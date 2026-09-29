"""The single asynchronous DreamHandoff inference engine."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any

import torch
from lerobot.policies.rtc import LatencyTracker
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.rollout import InferenceEngine
from torch import Tensor

from dream_handoff.r2dreamer import R2DreamerRuntime, R2State

from .bank import CandidateBank, GeneratedCandidateActions
from .config import DreamHandoffInferenceConfig
from .sampling import SmolVLACandidateSampler, configure_rtc_prefix_guidance
from .selector import CandidateSelector, SelectionResult

logger = logging.getLogger(__name__)

_WORKER_WAIT_SECONDS = 0.05
_WORKER_JOIN_SECONDS = 3.0
_RESET_WAIT_SECONDS = 30.0

EventSink = Callable[[str, Mapping[str, Any]], None]


def _clone_tensor_values(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if torch.is_tensor(value) else value for key, value in values.items()
    }


@dataclass(frozen=True)
class _BankRequest:
    policy_observation: dict[str, Any]
    action_count: int
    request_id: int
    epoch: int
    launched_at: float
    cuda_input_ready: Any | None
    guidance_mode: str
    predicted_delay: int
    request_phase: int
    prefix_actions: Tensor | None
    old_bank_origin_action_count: int


@dataclass(frozen=True)
class _ReadyBank:
    request: _BankRequest
    generated: GeneratedCandidateActions
    ready_action_count: int


class DreamHandoffInferenceEngine(InferenceEngine):
    """Generate candidate actions in one worker and dream them at activation."""

    def __init__(
        self,
        *,
        config: DreamHandoffInferenceConfig,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        r2_runtime: R2DreamerRuntime,
        dataset_features: dict[str, Any],
        ordered_action_keys: list[str],
        task: str,
        fps: float,
        device: str | torch.device | None,
        robot_type: str,
        shutdown_event: Event | None = None,
        event_sink: EventSink | None = None,
        candidate_sampler: SmolVLACandidateSampler | None = None,
        monotonic_clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self._config = config
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._r2 = r2_runtime
        self._dataset_features = dataset_features
        self._ordered_action_keys = list(ordered_action_keys)
        self._task = task
        self._fps = float(fps)
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        self._global_shutdown = shutdown_event
        self._event_sink = event_sink
        self._clock = monotonic_clock

        self._validate_compatibility()
        if config.async_bank_guidance == "static_rtc":
            configure_rtc_prefix_guidance(policy)
        self._sampler = candidate_sampler or SmolVLACandidateSampler(
            policy,
            preprocessor,
            postprocessor,
            device=str(self._device),
        )
        self._selector = CandidateSelector(
            config.selector_mode,
            hysteresis_tau=config.hysteresis_tau,
            seed=config.phase0_seed,
        )
        self._latency_tracker = LatencyTracker()

        self._state_lock = Lock()
        self._build_idle = Event()
        self._build_idle.set()
        self._request_available = Event()
        self._policy_active = Event()
        self._shutdown = Event()
        self._worker_failed = Event()
        self._worker_thread: Thread | None = None
        self._worker_error: BaseException | None = None

        self._pending: _BankRequest | None = None
        self._in_flight: _BankRequest | None = None
        self._ready_bank: _ReadyBank | None = None
        self._epoch = 0
        self._next_request_id = 0
        self._active_bank: CandidateBank | None = None
        self._actions_served = 0
        self._previous_action: Tensor | None = None
        self._imagination_compiled = False
        self._last_selection: SelectionResult | None = None

    def _validate_compatibility(self) -> None:
        policy_config = getattr(self._policy, "config", None)
        chunk_size = int(getattr(policy_config, "chunk_size", 0))
        max_action_dim = int(getattr(policy_config, "max_action_dim", 0))
        if chunk_size <= 0 or max_action_dim <= 0:
            raise ValueError("policy must expose positive SmolVLA chunk_size and max_action_dim")
        if self._config.max_bank_age > chunk_size:
            raise ValueError("max_bank_age cannot exceed the SmolVLA chunk size")

        action_names = self._dataset_features.get("action", {}).get("names")
        if action_names is None:
            raise ValueError("dataset features must define ordered action names")
        action_names = tuple(action_names)
        if tuple(self._r2.action_names) != action_names:
            raise ValueError("R2 canonical action names do not match policy/control action order")
        if len(action_names) > max_action_dim:
            raise ValueError("canonical action dimension exceeds SmolVLA max_action_dim")
        if len(self._ordered_action_keys) != len(action_names) or set(
            self._ordered_action_keys
        ) != set(action_names):
            raise ValueError("ordered_action_keys must be a permutation of canonical action names")
        missing = sorted(set(self._r2.observation_keys) - set(self._dataset_features))
        if missing:
            raise ValueError(f"R2 checkpoint requires rollout observation features {missing}")

    @property
    def ready(self) -> bool:
        return True

    @property
    def failed(self) -> bool:
        return self._worker_failed.is_set()

    def start(self) -> None:
        """Warm fixed-shape imagination once, then start one generation worker."""
        if self._config.compile_world_model_imagination and not self._imagination_compiled:
            self._r2.compile_imagination(
                self._config.num_candidates,
                min(self._config.max_bank_age, int(self._policy.config.chunk_size)),
            )
            self._imagination_compiled = True
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._shutdown.clear()
        self._worker_failed.clear()
        self._worker_error = None
        self._worker_thread = Thread(
            target=self._worker_loop,
            daemon=True,
            name="DreamHandoffBankWorker",
        )
        self._worker_thread.start()

    def stop(self) -> None:
        """Invalidate all work and stop the generation worker."""
        with self._state_lock:
            self._epoch += 1
            pending = self._pending
            ready = self._ready_bank
            self._pending = None
            if self._in_flight is pending:
                self._in_flight = None
            self._ready_bank = None
            self._active_bank = None
            self._selector.reset_bank()
        if pending is not None:
            self._emit_stale(pending, "engine stopped before generation completed")
        if ready is not None:
            self._emit_stale(ready.request, "engine stopped before activation")
        self._shutdown.set()
        self._policy_active.clear()
        self._request_available.set()
        worker = self._worker_thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=_WORKER_JOIN_SECONDS)
            if worker.is_alive():
                logger.warning(
                    "DreamHandoff worker did not stop within %.1fs", _WORKER_JOIN_SECONDS
                )
            else:
                self._worker_thread = None

    def pause(self) -> None:
        self._policy_active.clear()

    def resume(self) -> None:
        self._policy_active.set()
        with self._state_lock:
            pending = self._pending is not None
        if pending:
            self._request_available.set()

    def reset(self) -> None:
        """Quiesce generation and clear episode state without resetting rollout state."""
        with self._state_lock:
            self._epoch += 1
            pending = self._pending
            ready = self._ready_bank
            self._pending = None
            if self._in_flight is pending:
                self._in_flight = None
            self._ready_bank = None
            self._request_available.clear()
        if pending is not None:
            self._emit_stale(pending, "episode reset before generation completed")
        if ready is not None:
            self._emit_stale(ready.request, "episode reset before activation")
        if not self._build_idle.wait(timeout=_RESET_WAIT_SECONDS):
            raise RuntimeError("DreamHandoff worker did not quiesce before reset timeout")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        self._r2.reset()
        with self._state_lock:
            self._active_bank = None
            self._actions_served = 0
            self._previous_action = None
            self._last_selection = None
            self._selector.reset_bank()
        self._emit("episode_started", {"epoch": self._epoch})

    def get_action(self, obs_frame: dict | None) -> Tensor | None:
        """Advance R2 once, activate/request banks, select, and serve one action."""
        if obs_frame is None:
            return None
        self._raise_worker_error()
        prepared = prepare_observation_for_inference(
            copy(obs_frame), self._device, self._task, self._robot_type
        )
        snapshot = _clone_tensor_values(prepared) if self._snapshot_may_be_needed() else None
        with torch.inference_mode():
            live_state = self._r2.observe(prepared, self._previous_action)

        if self._active_bank is None:
            if snapshot is None:  # pragma: no cover - defensive
                snapshot = _clone_tensor_values(prepared)
            self._install_initial_bank(
                snapshot,
                live_state,
                exact_observation=obs_frame,
            )
        else:
            self._activate_ready_if_available(live_state, activation_observation=obs_frame)

        if self._should_request_refill():
            if snapshot is not None:
                self._launch_request(snapshot, exact_observation=obs_frame)
            else:
                logger.debug("Deferring refill until a fresh request observation is available")

        bank = self._active_bank
        if bank is None:  # pragma: no cover - initial generation raises first
            raise RuntimeError("DreamHandoff has no active candidate bank")
        with self._state_lock:
            action_count = self._actions_served
        phase = action_count - bank.origin_action_count
        if phase < 0:
            raise RuntimeError("active candidate bank originates in the future")
        if phase >= bank.horizon:
            raise RuntimeError("active DreamHandoff candidate bank expired before replacement")

        with torch.inference_mode():
            live_feature = self._r2.matching_feature(live_state)
            selection = self._selector.select(
                live_feature,
                bank.dreamed_matching_features[:, phase],
                phase=phase,
            )
            control_action = bank.control_actions[selection.index, phase].detach().clone()
            policy_action = bank.policy_actions[selection.index, phase].detach().clone()
        if control_action.ndim != 1 or not bool(torch.isfinite(control_action).all()):
            raise ValueError("selected canonical control action must be finite with shape [A]")
        if control_action.shape[0] != len(self._r2.action_names):
            raise ValueError("selected control action dimension does not match R2 action metadata")
        ordered_action = self._ordered_robot_action(control_action)
        self._previous_action = control_action
        self._last_selection = selection
        with self._state_lock:
            self._actions_served += 1
        self._emit(
            "action_served",
            {
                "action_count": action_count,
                "bank_origin": bank.origin_action_count,
                "phase": phase,
                "selected_candidate_index": selection.index,
                "challenger_candidate_index": selection.challenger_index,
                "incumbent_candidate_index": selection.incumbent_index,
                "hysteresis_advantage": selection.advantage,
                "candidate_distances": selection.distances,
                "candidate_switched": selection.switched,
                "selection_rule": selection.rule,
                "selected_policy_action": policy_action,
                "control_action": control_action,
                "live_world_model_stoch": live_state.stoch,
                "live_world_model_deter": live_state.deter,
            },
        )
        return ordered_action

    def _generate(self, request: _BankRequest) -> GeneratedCandidateActions:
        policy, control = self._sampler.sample_candidate_bank(
            request.policy_observation,
            self._config.num_candidates,
            prefix_actions=request.prefix_actions,
            inference_delay=(
                request.predicted_delay if request.prefix_actions is not None else None
            ),
            execution_horizon=(
                self._config.guidance_horizon if request.prefix_actions is not None else None
            ),
        )
        sampled_noise = getattr(self._sampler, "last_noise", None)
        generated = GeneratedCandidateActions(
            policy,
            control,
            request.action_count,
            sampled_noise=sampled_noise,
        )
        if generated.num_candidates != self._config.num_candidates:
            raise ValueError("SmolVLA returned an unexpected candidate count")
        if generated.control_actions.shape[-1] != len(self._r2.action_names):
            raise ValueError("postprocessed control action dimension does not match R2")
        return generated

    def _activate(
        self,
        generated: GeneratedCandidateActions,
        live_state: R2State,
        *,
        request_id: int,
        epoch: int,
        activation_count: int,
        actual_elapsed: int,
        activation_observation: Mapping[str, Any],
    ) -> CandidateBank:
        usable = min(generated.horizon - actual_elapsed, self._config.max_bank_age)
        if usable <= 0:
            raise RuntimeError("candidate generation finished too late to leave a usable bank")
        policy = generated.policy_actions[:, actual_elapsed : actual_elapsed + usable]
        control = generated.control_actions[:, actual_elapsed : actual_elapsed + usable]
        with torch.inference_mode():
            trajectory = self._r2.imagine_states(live_state, control)
            dreamed_features = self._r2.matching_features(trajectory)
        bank = CandidateBank(policy, control, dreamed_features, activation_count)
        self._active_bank = bank
        self._selector.reset_bank()
        self._emit(
            "bank_activated",
            {
                "request_action_count": generated.request_action_count,
                "request_id": request_id,
                "epoch": epoch,
                "activation_action_count": activation_count,
                "actual_elapsed": actual_elapsed,
                "usable_horizon": bank.horizon,
                "bank_origin_action_count": bank.origin_action_count,
                "bank_policy_actions": bank.policy_actions,
                "bank_control_actions": bank.control_actions,
                "bank_dreamed_matching_features": bank.dreamed_matching_features,
                "sampled_noise": (
                    None
                    if generated.sampled_noise is None
                    else generated.sampled_noise[:, actual_elapsed : actual_elapsed + usable]
                ),
                "activation_observation": activation_observation,
            },
        )
        return bank

    def _install_initial_bank(
        self,
        observation: dict[str, Any],
        live_state: R2State,
        *,
        exact_observation: Mapping[str, Any],
    ) -> None:
        request = self._make_request(observation, old_bank=None)
        self._emit_request(request, exact_observation=exact_observation)
        generated = self._generate(request)
        self._emit_ready(request, generated, request.action_count)
        if self._config.async_bank_guidance == "static_rtc":
            started = self._clock()
            prefix = generated.policy_actions[:, : self._config.guidance_horizon].detach().clone()
            warmup = _BankRequest(
                policy_observation=_clone_tensor_values(request.policy_observation),
                action_count=request.action_count,
                request_id=request.request_id,
                epoch=request.epoch,
                launched_at=started,
                cuda_input_ready=None,
                guidance_mode="static_rtc",
                predicted_delay=0,
                request_phase=0,
                prefix_actions=prefix,
                old_bank_origin_action_count=request.old_bank_origin_action_count,
            )
            self._generate(warmup)
            self._latency_tracker.add(max(0.0, self._clock() - started))
        self._activate(
            generated,
            live_state,
            request_id=request.request_id,
            epoch=request.epoch,
            activation_count=request.action_count,
            actual_elapsed=0,
            activation_observation=exact_observation,
        )

    def _activate_ready_if_available(
        self,
        live_state: R2State,
        *,
        activation_observation: Mapping[str, Any],
    ) -> bool:
        with self._state_lock:
            ready = self._ready_bank
            if ready is None:
                return False
            self._ready_bank = None
            activation_count = self._actions_served
            current_epoch = self._epoch
        if ready.request.epoch != current_epoch:
            self._emit(
                "stale_generation",
                {
                    "request_id": ready.request.request_id,
                    "epoch": ready.request.epoch,
                    "request_action_count": ready.request.action_count,
                    "stale_reason": "request epoch no longer active",
                },
            )
            return False
        self._activate(
            ready.generated,
            live_state,
            request_id=ready.request.request_id,
            epoch=ready.request.epoch,
            activation_count=activation_count,
            actual_elapsed=activation_count - ready.request.action_count,
            activation_observation=activation_observation,
        )
        return True

    def _snapshot_may_be_needed(self) -> bool:
        with self._state_lock:
            bank = self._active_bank
            if bank is None or self._ready_bank is not None:
                return True
            phase = self._actions_served - bank.origin_action_count
            remaining = bank.horizon - phase
            if self._pending is not None or self._in_flight is not None:
                return remaining <= 0
            return remaining <= self._config.bank_refill_threshold

    def _should_request_refill(self) -> bool:
        with self._state_lock:
            if (
                self._pending is not None
                or self._in_flight is not None
                or self._ready_bank is not None
            ):
                return False
            bank = self._active_bank
            if bank is None:
                return False
            phase = self._actions_served - bank.origin_action_count
            return bank.horizon - phase <= self._config.bank_refill_threshold

    def _make_request(
        self,
        observation: dict[str, Any],
        *,
        old_bank: CandidateBank | None,
    ) -> _BankRequest:
        cuda_event = None
        if self._device.type == "cuda":
            cuda_event = torch.cuda.Event()
        with self._state_lock:
            action_count = self._actions_served
            request_id = self._next_request_id
            self._next_request_id += 1
            epoch = self._epoch
            predicted_seconds = self._latency_tracker.max() or 0.0
            predicted_delay = math.ceil(predicted_seconds * self._fps) if predicted_seconds else 0
            request_phase = -1 if old_bank is None else action_count - old_bank.origin_action_count
            old_bank_origin = -1 if old_bank is None else old_bank.origin_action_count
            prefix = None
            mode = "plain"
            if old_bank is not None and self._config.async_bank_guidance == "static_rtc":
                end = request_phase + self._config.guidance_horizon
                if request_phase < 0 or end > old_bank.horizon:
                    raise RuntimeError("static_rtc guidance horizon exceeds the active bank")
                prefix = old_bank.policy_actions[:, request_phase:end].detach().clone()
                mode = "static_rtc"
        if cuda_event is not None:
            cuda_event.record(torch.cuda.current_stream(self._device))
        return _BankRequest(
            policy_observation=observation,
            action_count=action_count,
            request_id=request_id,
            epoch=epoch,
            launched_at=self._clock(),
            cuda_input_ready=cuda_event,
            guidance_mode=mode,
            predicted_delay=predicted_delay,
            request_phase=request_phase,
            prefix_actions=prefix,
            old_bank_origin_action_count=old_bank_origin,
        )

    def _launch_request(
        self,
        observation: dict[str, Any],
        *,
        exact_observation: Mapping[str, Any],
    ) -> None:
        request = self._make_request(observation, old_bank=self._active_bank)
        with self._state_lock:
            if request.epoch != self._epoch or request.action_count != self._actions_served:
                return
            if (
                self._pending is not None
                or self._in_flight is not None
                or self._ready_bank is not None
            ):
                return
            self._pending = request
            self._in_flight = request
        self._emit_request(request, exact_observation=exact_observation)
        self._request_available.set()

    def _worker_loop(self) -> None:
        try:
            self._worker_loop_body()
        except Exception as exc:  # pragma: no cover - catastrophic loop failure
            self._worker_error = exc
            self._worker_failed.set()
            if self._global_shutdown is not None:
                self._global_shutdown.set()
            self._emit_worker_failure(exc)

    def _worker_loop_body(self) -> None:
        stream = torch.cuda.Stream(device=self._device) if self._device.type == "cuda" else None
        while not self._shutdown.is_set():
            if not self._request_available.wait(timeout=_WORKER_WAIT_SECONDS):
                continue
            if self._shutdown.is_set():
                break
            if not self._policy_active.is_set():
                self._policy_active.wait(timeout=_WORKER_WAIT_SECONDS)
                continue
            with self._state_lock:
                request = self._pending
                self._pending = None
                self._request_available.clear()
                if request is not None:
                    self._build_idle.clear()
            if request is None:
                continue
            try:
                generated = self._run_worker_generation(request, stream)
                ready_at = self._clock()
                latency = max(0.0, ready_at - request.launched_at)
                with self._state_lock:
                    self._latency_tracker.add(latency)
                    ready_count = self._actions_served
                    valid = request.epoch == self._epoch and not self._shutdown.is_set()
                    if self._in_flight is request:
                        self._in_flight = None
                    if valid:
                        self._ready_bank = _ReadyBank(request, generated, ready_count)
                if valid:
                    self._emit_ready(request, generated, ready_count)
                else:
                    self._emit_stale(
                        request,
                        "request completed after reset or shutdown",
                        generated=generated,
                        ready_action_count=ready_count,
                    )
            except Exception as exc:
                with self._state_lock:
                    if self._in_flight is request:
                        self._in_flight = None
                    obsolete = request.epoch != self._epoch
                if not obsolete and not self._shutdown.is_set():
                    self._worker_error = exc
                    self._worker_failed.set()
                    if self._global_shutdown is not None:
                        self._global_shutdown.set()
                    self._emit_worker_failure(exc, request=request)
                    return
            finally:
                self._build_idle.set()

    def _run_worker_generation(
        self,
        request: _BankRequest,
        stream: Any | None,
    ) -> GeneratedCandidateActions:
        if stream is None:
            return self._generate(request)
        with torch.cuda.stream(stream):
            if request.cuda_input_ready is not None:
                stream.wait_event(request.cuda_input_ready)
            try:
                generated = self._generate(request)
            finally:
                stream.synchronize()
        return generated

    def _raise_worker_error(self) -> None:
        if not self._worker_failed.is_set():
            return
        if self._worker_error is None:  # pragma: no cover - defensive
            raise RuntimeError("DreamHandoff bank worker failed")
        raise RuntimeError("DreamHandoff bank worker failed") from self._worker_error

    def _ordered_robot_action(self, action: Tensor) -> Tensor:
        action_cpu = action.detach().cpu()
        action_dict = make_robot_action(action_cpu, self._dataset_features)
        try:
            values = [action_dict[key] for key in self._ordered_action_keys]
        except KeyError as exc:
            raise ValueError(
                f"ordered action key is absent from policy output: {exc.args[0]}"
            ) from exc
        return torch.tensor(values, dtype=action_cpu.dtype)

    def _emit(self, event: str, values: Mapping[str, Any]) -> None:
        if self._event_sink is not None:
            self._event_sink(event, values)

    def _emit_request(
        self,
        request: _BankRequest,
        *,
        exact_observation: Mapping[str, Any],
    ) -> None:
        self._emit(
            "bank_requested",
            {
                "request_id": request.request_id,
                "epoch": request.epoch,
                "request_action_count": request.action_count,
                "request_phase": request.request_phase,
                "old_bank_origin_action_count": request.old_bank_origin_action_count,
                "guidance_mode": request.guidance_mode,
                "predicted_delay": request.predicted_delay,
                "guidance_horizon": (
                    self._config.guidance_horizon if request.guidance_mode == "static_rtc" else -1
                ),
                "prefix_actions": request.prefix_actions,
                "request_observation": exact_observation,
            },
        )

    def _emit_ready(
        self,
        request: _BankRequest,
        generated: GeneratedCandidateActions,
        ready_action_count: int,
    ) -> None:
        self._emit(
            "generation_ready",
            {
                "request_id": request.request_id,
                "epoch": request.epoch,
                "ready_action_count": ready_action_count,
                "generated_policy_actions": generated.policy_actions,
                "generated_control_actions": generated.control_actions,
                "sampled_noise": generated.sampled_noise,
            },
        )

    def _emit_stale(
        self,
        request: _BankRequest,
        reason: str,
        *,
        generated: GeneratedCandidateActions | None = None,
        ready_action_count: int | None = None,
    ) -> None:
        values: dict[str, Any] = {
            "request_id": request.request_id,
            "epoch": request.epoch,
            "request_action_count": request.action_count,
            "stale_reason": reason,
        }
        if generated is not None:
            if ready_action_count is None:  # pragma: no cover - defensive
                raise ValueError("completed stale generation requires its ready action count")
            values.update(
                ready_action_count=ready_action_count,
                generated_policy_actions=generated.policy_actions,
                generated_control_actions=generated.control_actions,
                sampled_noise=generated.sampled_noise,
            )
        self._emit("stale_generation", values)

    def _emit_worker_failure(
        self,
        error: BaseException,
        *,
        request: _BankRequest | None = None,
    ) -> None:
        try:
            self._emit(
                "worker_failure",
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "request_id": None if request is None else request.request_id,
                    "epoch": None if request is None else request.epoch,
                },
            )
        except Exception:
            logger.exception("DreamHandoff worker-failure event hook also failed")

    @property
    def candidate_bank(self) -> CandidateBank | None:
        return self._active_bank

    @property
    def actions_served(self) -> int:
        with self._state_lock:
            return self._actions_served

    @property
    def previous_action(self) -> Tensor | None:
        return None if self._previous_action is None else self._previous_action.detach().clone()

    @property
    def last_selection(self) -> SelectionResult | None:
        return self._last_selection

    @property
    def request_in_flight(self) -> bool:
        with self._state_lock:
            return self._pending is not None or self._in_flight is not None

    @property
    def has_ready_bank(self) -> bool:
        with self._state_lock:
            return self._ready_bank is not None
