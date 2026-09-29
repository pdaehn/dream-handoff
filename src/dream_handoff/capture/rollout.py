"""Observational rollout capture and transparent LeRobot boundary wrappers."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from .schema import DIAGNOSTIC_FORMAT_VERSION, DREAMHANDOFF_DIAGNOSTIC_FORMAT

logger = logging.getLogger(__name__)


def _copy_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == object:
        raise ValueError("diagnostic values cannot use object dtype")
    return np.ascontiguousarray(array).copy()


def _state_component(value: Any) -> np.ndarray:
    array = _copy_array(value)
    if array.shape[:1] == (1,):
        array = array[0]
    return array


def _pad_phases(value: np.ndarray, horizon: int) -> np.ndarray:
    if value.shape[1] == horizon:
        return value
    padded = np.full(
        (value.shape[0], horizon, *value.shape[2:]),
        np.nan,
        dtype=value.dtype,
    )
    padded[:, : value.shape[1]] = value
    return padded


def _stack_or_empty(
    records: list[dict[str, Any]],
    name: str,
    *,
    dtype: Any,
    empty_shape: tuple[int, ...] = (0,),
) -> np.ndarray:
    if not records:
        return np.empty(empty_shape, dtype=dtype)
    return np.stack([record[name] for record in records]).astype(dtype, copy=False)


class RolloutCapture:
    """Buffer format-16 factual events and serialize only after rollout execution."""

    def __init__(self, output_path: str | Path) -> None:
        path = Path(output_path)
        self.output_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        if self.output_path.exists():
            raise FileExistsError(f"capture destination already exists: {self.output_path}")
        self._lock = Lock()
        self._started_at = time.perf_counter()
        self._created_at = datetime.now(UTC).isoformat()
        self._saved_path: Path | None = None
        self._configured = False
        self._action_keys: tuple[str, ...] = ()
        self._fps = 0.0
        self._interpolation_multiplier = 1
        self._policy_chunk_size = 0
        self._max_action_dim = 0
        self._num_candidates = 0
        self._bank_refill_threshold = -1
        self._guidance_horizon = -1
        self._guidance_mode = "plain"
        self._selector_mode = ""
        self._hysteresis_tau: float | None = None
        self._phase0_seed = 0
        self._task = ""
        self._robot_type = ""

        self._episode_id = -1
        self._first_action_in_episode = True
        self._commands_remaining = 0
        self._current_action: dict[str, Any] | None = None
        self._latest_measured: np.ndarray | None = None
        self._processor_input: np.ndarray | None = None
        self._processor_output: np.ndarray | None = None
        self._commands: list[dict[str, Any]] = []
        self._banks: list[dict[str, Any]] = []
        self._requests: dict[tuple[int, int], dict[str, Any]] = {}
        self._request_order: list[tuple[int, int]] = []
        self._sparse_observations: list[dict[str, Any]] = []
        self._worker_failures: list[dict[str, Any]] = []
        self._capture_errors: list[str] = []
        self._rollout_failure: dict[str, str] | None = None

    @property
    def saved_path(self) -> Path | None:
        return self._saved_path

    def configure(
        self,
        *,
        action_keys: list[str] | tuple[str, ...],
        fps: float,
        interpolation_multiplier: int,
        policy_chunk_size: int,
        max_action_dim: int,
        num_candidates: int,
        bank_refill_threshold: int,
        guidance_horizon: int,
        guidance_mode: str,
        selector_mode: str,
        hysteresis_tau: float | None,
        phase0_seed: int,
        task: str,
        robot_type: str,
    ) -> None:
        """Pin run metadata before stock strategy setup starts the engine."""

        keys = tuple(str(key) for key in action_keys)
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("capture requires unique ordered action keys")
        if fps <= 0 or interpolation_multiplier < 1:
            raise ValueError("capture requires positive fps and interpolation multiplier")
        if policy_chunk_size <= 0 or max_action_dim <= 0 or num_candidates <= 0:
            raise ValueError("capture requires positive policy and candidate dimensions")
        with self._lock:
            self._action_keys = keys
            self._fps = float(fps)
            self._interpolation_multiplier = int(interpolation_multiplier)
            self._policy_chunk_size = int(policy_chunk_size)
            self._max_action_dim = int(max_action_dim)
            self._num_candidates = int(num_candidates)
            self._bank_refill_threshold = int(bank_refill_threshold)
            self._guidance_horizon = int(guidance_horizon)
            self._guidance_mode = str(guidance_mode)
            self._selector_mode = str(selector_mode)
            self._hysteresis_tau = None if hysteresis_tau is None else float(hysteresis_tau)
            self._phase0_seed = int(phase0_seed)
            self._task = str(task)
            self._robot_type = str(robot_type)
            self._configured = True

    def event_sink(self, event: str, values: Mapping[str, Any]) -> None:
        """Consume one WP3 factual event without allowing capture failure into control."""

        try:
            with self._lock:
                self._record_event(event, values)
        except Exception as exc:  # diagnostics must not change controller behavior
            self._remember_capture_error(event, exc)

    def record_rollout_failure(self, error: BaseException) -> None:
        with self._lock:
            if self._rollout_failure is None:
                self._rollout_failure = {
                    "type": type(error).__name__,
                    "message": str(error),
                }

    def record_observation(self, observation: Mapping[str, Any]) -> None:
        try:
            measured = self._action_vector(observation)
            with self._lock:
                self._latest_measured = measured
        except Exception as exc:
            self._remember_capture_error("robot_observation", exc)

    def record_action_processor(self, processor_input: Any, processor_output: Any) -> None:
        try:
            input_vector = self._action_vector(processor_input)
            output_vector = self._action_vector(processor_output)
            with self._lock:
                if self._commands_remaining > 0:
                    self._processor_input = input_vector
                    self._processor_output = output_vector
        except Exception as exc:
            self._remember_capture_error("action_processor", exc)

    def record_command_sent(
        self,
        command: Any,
        *,
        dataset_episode_index: int | None,
        dataset_frame_index: int | None,
    ) -> None:
        try:
            command_vector = self._action_vector(command)
            with self._lock:
                if self._commands_remaining <= 0 or self._current_action is None:
                    return
                action_dim = len(self._action_keys)
                measured = (
                    self._latest_measured.copy()
                    if self._latest_measured is not None
                    else np.full(action_dim, np.nan, dtype=np.float32)
                )
                record = dict(self._current_action)
                record.update(
                    timestamp=time.perf_counter() - self._started_at,
                    wall_timestamp=time.time(),
                    command_step=len(self._commands),
                    dataset_episode_index=(
                        -1 if dataset_episode_index is None else int(dataset_episode_index)
                    ),
                    dataset_frame_index=(
                        -1 if dataset_frame_index is None else int(dataset_frame_index)
                    ),
                    interpolated_action=(
                        self._processor_input.copy()
                        if self._processor_input is not None
                        else record["final_executed_action"].copy()
                    ),
                    action_processor_output=(
                        self._processor_output.copy()
                        if self._processor_output is not None
                        else command_vector.copy()
                    ),
                    command_sent_to_robot=command_vector,
                    measured_joint_position=measured,
                )
                self._commands.append(record)
                self._commands_remaining -= 1
        except Exception as exc:
            self._remember_capture_error("robot_command", exc)

    def save(self) -> Path:
        """Write one compressed NPZ; callers invoke this after strategy teardown."""

        with self._lock:
            if self._saved_path is not None:
                return self._saved_path
            if not self._configured:
                raise RuntimeError("capture was never configured from a rollout context")
            commands = list(self._commands)
            banks = list(self._banks)
            requests = [dict(self._requests[key]) for key in self._request_order]
            sparse = list(self._sparse_observations)
            metadata = self._metadata(requests, sparse)
            arrays = self._serialize(commands, banks, requests, sparse, metadata)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.output_path, **arrays)
        with self._lock:
            self._saved_path = self.output_path
        return self.output_path

    def _record_event(self, event: str, values: Mapping[str, Any]) -> None:
        if event == "episode_started":
            self._episode_id += 1
            self._first_action_in_episode = True
            self._commands_remaining = 0
            self._current_action = None
            self._latest_measured = None
            self._processor_input = None
            self._processor_output = None
            return
        if event == "bank_requested":
            self._record_request(values)
            return
        if event == "generation_ready":
            self._record_ready(values)
            return
        if event == "bank_activated":
            self._record_activation(values)
            return
        if event == "stale_generation":
            self._record_stale(values)
            return
        if event == "action_served":
            self._record_action(values)
            return
        if event == "worker_failure":
            self._worker_failures.append(dict(values))

    def _record_request(self, values: Mapping[str, Any]) -> None:
        key = (int(values["epoch"]), int(values["request_id"]))
        now = time.perf_counter()
        record = {
            "episode_id": max(self._episode_id, 0),
            "request_id": key[1],
            "generation": key[0],
            "origin_action_count": int(values["request_action_count"]),
            "old_bank_origin_action_count": int(values["old_bank_origin_action_count"]),
            "request_phase": int(values["request_phase"]),
            "guidance_mode": str(values["guidance_mode"]),
            "guidance_horizon": int(values["guidance_horizon"]),
            "predicted_delay": int(values["predicted_delay"]),
            "prefix_actions": (
                None
                if values.get("prefix_actions") is None
                else _copy_array(values["prefix_actions"])
            ),
            "launch_perf_time": now,
            "launch_wall_time": time.time(),
            "ready_perf_time": None,
            "ready_wall_time": -1.0,
            "ready_action_count": -1,
            "generated_policy_actions": None,
            "generated_control_actions": None,
            "generation_noise": None,
            "activation_action_count": -1,
            "actual_elapsed": -1,
            "cropped_bank_horizon": -1,
            "stale": False,
            "stale_reason": "",
        }
        self._requests[key] = record
        self._request_order.append(key)
        self._append_sparse(
            record,
            event_role="request_origin",
            action_count=record["origin_action_count"],
            activation_action_count=-1,
            observation=values["request_observation"],
        )

    def _record_ready(self, values: Mapping[str, Any]) -> None:
        key = (int(values["epoch"]), int(values["request_id"]))
        record = self._requests.get(key)
        if record is None:
            raise ValueError(f"generation_ready has no request event: {key}")
        record["ready_perf_time"] = time.perf_counter()
        record["ready_wall_time"] = time.time()
        record["ready_action_count"] = int(values["ready_action_count"])
        record["generated_policy_actions"] = _copy_array(values["generated_policy_actions"])
        record["generated_control_actions"] = _copy_array(values["generated_control_actions"])
        noise = values.get("sampled_noise")
        record["generation_noise"] = None if noise is None else _copy_array(noise)

    def _record_activation(self, values: Mapping[str, Any]) -> None:
        key = (int(values["epoch"]), int(values["request_id"]))
        request = self._requests.get(key)
        if request is None:
            raise ValueError(f"bank_activated has no request event: {key}")
        activation_count = int(values["activation_action_count"])
        request["activation_action_count"] = activation_count
        request["actual_elapsed"] = int(values["actual_elapsed"])
        request["cropped_bank_horizon"] = int(values["usable_horizon"])
        self._banks.append(
            {
                "episode_id": request["episode_id"],
                "origin_action_count": int(values["bank_origin_action_count"]),
                "request_id": request["request_id"],
                "generation": request["generation"],
                "policy_actions": _copy_array(values["bank_policy_actions"]),
                "control_actions": _copy_array(values["bank_control_actions"]),
                "dreamed_matching_features": _copy_array(values["bank_dreamed_matching_features"]),
                "sampled_noise": (
                    None
                    if values.get("sampled_noise") is None
                    else _copy_array(values["sampled_noise"])
                ),
                "horizon": int(values["usable_horizon"]),
            }
        )
        self._append_sparse(
            request,
            event_role="activation",
            action_count=activation_count,
            activation_action_count=activation_count,
            observation=values["activation_observation"],
        )

    def _record_stale(self, values: Mapping[str, Any]) -> None:
        key = (int(values["epoch"]), int(values["request_id"]))
        request = self._requests.get(key)
        if request is None:
            return
        if values.get("generated_policy_actions") is not None:
            self._record_ready(values)
        request["stale"] = True
        request["stale_reason"] = str(values.get("stale_reason", "stale generation"))

    def _record_action(self, values: Mapping[str, Any]) -> None:
        control_action = _copy_array(values["control_action"]).reshape(-1)
        policy_action = _copy_array(values["selected_policy_action"]).reshape(-1)
        action_count = int(values["action_count"])
        self._current_action = {
            "control_step": action_count,
            "episode_id": max(self._episode_id, 0),
            "bank_origin_action_count": int(values["bank_origin"]),
            "phase": int(values["phase"]),
            "selected_candidate_index": int(values["selected_candidate_index"]),
            "challenger_candidate_index": int(values["challenger_candidate_index"]),
            "incumbent_candidate_index": (
                -1
                if values["incumbent_candidate_index"] is None
                else int(values["incumbent_candidate_index"])
            ),
            "hysteresis_advantage": (
                np.float32(np.nan)
                if values["hysteresis_advantage"] is None
                else np.float32(values["hysteresis_advantage"])
            ),
            "candidate_distances": _copy_array(values["candidate_distances"]).reshape(-1),
            "live_world_model_stoch": _state_component(values["live_world_model_stoch"]),
            "live_world_model_deter": _state_component(values["live_world_model_deter"]),
            "candidate_switched": bool(values["candidate_switched"]),
            "selection_rule": str(values["selection_rule"]),
            "selected_policy_action": policy_action,
            "selected_control_action": control_action,
            "final_executed_policy_action": policy_action.copy(),
            "final_executed_action": control_action.copy(),
        }
        self._commands_remaining = (
            1 if self._first_action_in_episode else self._interpolation_multiplier
        )
        self._first_action_in_episode = False
        self._processor_input = None
        self._processor_output = None

    def _append_sparse(
        self,
        request: Mapping[str, Any],
        *,
        event_role: str,
        action_count: int,
        activation_action_count: int,
        observation: Mapping[str, Any],
    ) -> None:
        values = {str(key): _copy_array(value) for key, value in sorted(observation.items())}
        if not values:
            raise ValueError("sparse exact observation cannot be empty")
        self._sparse_observations.append(
            {
                "episode_id": int(request["episode_id"]),
                "request_id": int(request["request_id"]),
                "generation": int(request["generation"]),
                "origin_action_count": int(request["origin_action_count"]),
                "activation_action_count": int(activation_action_count),
                "action_count": int(action_count),
                "event_role": str(event_role),
                "values": values,
            }
        )

    def _action_vector(self, values: Any) -> np.ndarray:
        if not isinstance(values, Mapping):
            array = _copy_array(values).reshape(-1)
        else:
            array = np.asarray(
                [np.asarray(values[key]).item() for key in self._action_keys],
                dtype=np.float32,
            )
        if array.shape != (len(self._action_keys),):
            raise ValueError("diagnostic action does not match ordered action keys")
        return array.astype(np.float32, copy=True)

    def _remember_capture_error(self, source: str, error: Exception) -> None:
        message = f"{source}: {type(error).__name__}: {error}"
        with self._lock:
            self._capture_errors.append(message)
        logger.error("Capture observation failed without interrupting control: %s", message)

    def _metadata(
        self,
        requests: list[dict[str, Any]],
        sparse: list[dict[str, Any]],
    ) -> dict[str, Any]:
        sparse_keys: list[dict[str, Any]] = []
        if sparse:
            first = sparse[0]["values"]
            for index, (key, value) in enumerate(first.items()):
                sparse_keys.append(
                    {
                        "key": key,
                        "array": f"sparse_observation_value_{index}",
                        "dtype": str(value.dtype),
                        "shape_per_capture": list(value.shape),
                    }
                )
        complete_noise = bool(requests) and all(
            request["generation_noise"] is not None for request in requests
        )
        return {
            "format": DREAMHANDOFF_DIAGNOSTIC_FORMAT,
            "format_version": DIAGNOSTIC_FORMAT_VERSION,
            "created_at": self._created_at,
            "producer": "dream-handoff WP5 observational capture",
            "action_keys": list(self._action_keys),
            "fps": self._fps,
            "interpolation_multiplier": self._interpolation_multiplier,
            "policy_chunk_size": self._policy_chunk_size,
            "max_action_dim": self._max_action_dim,
            "controller": {
                "num_candidates": self._num_candidates,
                "bank_refill_threshold": self._bank_refill_threshold,
                "guidance_horizon": self._guidance_horizon,
                "async_bank_guidance": self._guidance_mode,
                "selector_mode": self._selector_mode,
                "hysteresis_tau": self._hysteresis_tau,
                "phase0_seed": self._phase0_seed,
            },
            "sparse_exact_observations": {
                "task": self._task,
                "robot_type": self._robot_type,
                "keys": sparse_keys,
                "roles": ["request_origin", "activation"],
                "capture_seam": (
                    "same already-acquired engine obs_frame; no additional robot or camera read"
                ),
            },
            "request_generation": {
                "noise": "uncropped explicit SmolVLA flow noise in request order",
                "generated_actions": (
                    "uncropped worker-ready policy/control chunks; activated banks are "
                    "stored separately after the factual elapsed-action crop"
                ),
            },
            "capture_complete": (
                self._rollout_failure is None
                and not self._capture_errors
                and complete_noise
                and all(
                    request["activation_action_count"] >= 0 or request["stale"]
                    for request in requests
                )
            ),
            "rollout_failure": self._rollout_failure,
            "worker_failures": self._worker_failures,
            "capture_errors": list(self._capture_errors),
            "omitted_derived_format16_fields": [
                "action_diversity_*",
                "dream_diversity_*",
                "counterfactual_same_phase_switch_*",
            ],
            "omitted_optional_format16_fields": ["candidate_bank_dream_seconds"],
        }

    def _serialize(
        self,
        commands: list[dict[str, Any]],
        banks: list[dict[str, Any]],
        requests: list[dict[str, Any]],
        sparse: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        action_dim = len(self._action_keys)
        command_scalar_types = {
            "timestamp": np.float64,
            "wall_timestamp": np.float64,
            "command_step": np.int64,
            "control_step": np.int64,
            "episode_id": np.int64,
            "dataset_episode_index": np.int64,
            "dataset_frame_index": np.int64,
            "bank_origin_action_count": np.int64,
            "phase": np.int64,
            "selected_candidate_index": np.int64,
            "challenger_candidate_index": np.int64,
            "incumbent_candidate_index": np.int64,
            "hysteresis_advantage": np.float32,
            "candidate_switched": np.bool_,
            "selection_rule": np.str_,
        }
        arrays = {
            name: np.asarray([record[name] for record in commands], dtype=dtype)
            for name, dtype in command_scalar_types.items()
        }
        arrays["candidate_distances"] = _stack_or_empty(
            commands,
            "candidate_distances",
            dtype=np.float32,
            empty_shape=(0, self._num_candidates),
        )
        arrays["live_world_model_stoch"] = _stack_or_empty(
            commands, "live_world_model_stoch", dtype=np.float32
        )
        arrays["live_world_model_deter"] = _stack_or_empty(
            commands, "live_world_model_deter", dtype=np.float32
        )
        for name in (
            "selected_policy_action",
            "selected_control_action",
            "final_executed_policy_action",
            "final_executed_action",
            "interpolated_action",
            "action_processor_output",
            "command_sent_to_robot",
            "measured_joint_position",
        ):
            arrays[name] = _stack_or_empty(
                commands, name, dtype=np.float32, empty_shape=(0, action_dim)
            )

        bank_horizon = max((record["horizon"] for record in banks), default=0)
        arrays["candidate_bank_episode_id"] = np.asarray(
            [record["episode_id"] for record in banks], dtype=np.int64
        )
        arrays["candidate_bank_origin_action_count"] = np.asarray(
            [record["origin_action_count"] for record in banks], dtype=np.int64
        )
        arrays["candidate_bank_request_id"] = np.asarray(
            [record["request_id"] for record in banks], dtype=np.int64
        )
        arrays["candidate_bank_generation"] = np.asarray(
            [record["generation"] for record in banks], dtype=np.int64
        )
        arrays["candidate_bank_horizon"] = np.asarray(
            [record["horizon"] for record in banks], dtype=np.int64
        )

        def bank_values(name: str, width: int) -> np.ndarray:
            if not banks:
                return np.empty((0, self._num_candidates, 0, width), dtype=np.float32)
            return np.stack([_pad_phases(record[name], bank_horizon) for record in banks]).astype(
                np.float32
            )

        arrays["candidate_bank_policy_actions"] = bank_values("policy_actions", action_dim)
        arrays["candidate_bank_control_actions"] = bank_values("control_actions", action_dim)
        dream_width = int(banks[0]["dreamed_matching_features"].shape[-1]) if banks else 0
        arrays["candidate_bank_dreamed_matching_features"] = bank_values(
            "dreamed_matching_features", dream_width
        )
        if banks:
            present_noise = [
                record["sampled_noise"] for record in banks if record["sampled_noise"] is not None
            ]
            noise_width = int(present_noise[0].shape[-1]) if present_noise else self._max_action_dim
            noise_rows = []
            for record in banks:
                noise = record["sampled_noise"]
                if noise is None:
                    noise = np.full(
                        (self._num_candidates, record["horizon"], noise_width),
                        np.nan,
                        dtype=np.float32,
                    )
                noise_rows.append(_pad_phases(noise, bank_horizon))
            arrays["candidate_bank_sampled_noise"] = np.stack(noise_rows).astype(np.float32)
        else:
            arrays["candidate_bank_sampled_noise"] = np.empty(
                (0, self._num_candidates, 0, self._max_action_dim), dtype=np.float32
            )

        request_values = self._serialize_requests(requests, action_dim)
        arrays.update(request_values)
        arrays.update(self._serialize_sparse(sparse, metadata))
        arrays["action_keys"] = np.asarray(self._action_keys)
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
        return arrays

    def _serialize_requests(
        self,
        requests: list[dict[str, Any]],
        action_dim: int,
    ) -> dict[str, np.ndarray]:
        now = time.perf_counter()
        for request in requests:
            if request["activation_action_count"] < 0 and not request["stale"]:
                request["stale"] = True
                request["stale_reason"] = "capture finalized before activation"
            ready_perf = request["ready_perf_time"]
            request["build_latency_seconds"] = (
                -1.0 if ready_perf is None else max(0.0, ready_perf - request["launch_perf_time"])
            )
            if ready_perf is None and request["stale"]:
                request["ready_perf_time"] = now
            request["actions_consumed_while_building"] = (
                -1
                if request["ready_action_count"] < 0
                else request["ready_action_count"] - request["origin_action_count"]
            )
            request["actual_consumed_delay_steps"] = (
                request["actual_elapsed"]
                if request["actual_elapsed"] >= 0
                else request["actions_consumed_while_building"]
            )
            latency = request["build_latency_seconds"]
            request["wall_clock_delay_steps"] = (
                -1 if latency < 0 else int(math.ceil(latency * self._fps))
            )
            request["delay_difference_steps"] = (
                -1
                if request["wall_clock_delay_steps"] < 0
                or request["actual_consumed_delay_steps"] < 0
                else request["wall_clock_delay_steps"] - request["actual_consumed_delay_steps"]
            )

        mapping = {
            "async_request_episode_id": ("episode_id", np.int64),
            "async_request_id": ("request_id", np.int64),
            "async_request_generation": ("generation", np.int64),
            "async_request_origin_action_count": ("origin_action_count", np.int64),
            "async_request_old_bank_origin_action_count": (
                "old_bank_origin_action_count",
                np.int64,
            ),
            "async_request_launch_wall_time": ("launch_wall_time", np.float64),
            "async_request_ready_wall_time": ("ready_wall_time", np.float64),
            "async_request_build_latency_seconds": ("build_latency_seconds", np.float64),
            "async_request_actions_consumed_while_building": (
                "actions_consumed_while_building",
                np.int64,
            ),
            "async_request_wall_clock_delay_steps": ("wall_clock_delay_steps", np.int64),
            "async_request_actual_consumed_delay_steps": (
                "actual_consumed_delay_steps",
                np.int64,
            ),
            "async_request_delay_difference_steps": ("delay_difference_steps", np.int64),
            "async_request_activation_action_count": ("activation_action_count", np.int64),
            "async_request_activation_phase": ("actual_elapsed", np.int64),
            "async_request_stale": ("stale", np.bool_),
            "async_request_stale_reason": ("stale_reason", np.str_),
            "async_request_rtc_guidance_mode": ("guidance_mode", np.str_),
            "async_request_rtc_guidance_horizon": ("guidance_horizon", np.int64),
            "async_request_rtc_request_phase": ("request_phase", np.int64),
            "async_request_bank_refill_threshold": (None, np.int64),
            "async_request_cropped_bank_horizon": ("cropped_bank_horizon", np.int64),
            "async_request_activation_reimagination_seconds": (None, np.float64),
            "async_request_predicted_inference_delay_steps": ("predicted_delay", np.int64),
            "async_request_rtc_predicted_inference_delay": (None, np.int64),
        }
        arrays: dict[str, np.ndarray] = {}
        for output, (source, dtype) in mapping.items():
            if source is not None:
                values = [request[source] for request in requests]
            elif output == "async_request_bank_refill_threshold":
                values = [self._bank_refill_threshold] * len(requests)
            elif output == "async_request_activation_reimagination_seconds":
                values = [-1.0] * len(requests)
            else:
                values = [
                    request["predicted_delay"] if request["guidance_mode"] == "static_rtc" else -1
                    for request in requests
                ]
            arrays[output] = np.asarray(values, dtype=dtype)

        noise_shape = (
            len(requests),
            self._num_candidates,
            self._policy_chunk_size,
            self._max_action_dim,
        )
        generation_noise = np.full(noise_shape, np.nan, dtype=np.float32)
        for index, request in enumerate(requests):
            value = request["generation_noise"]
            if value is None:
                continue
            if value.shape != noise_shape[1:]:
                raise ValueError(
                    "request generation noise does not match configured [N,H,D] dimensions"
                )
            generation_noise[index] = value
        arrays["async_request_generation_noise"] = generation_noise

        prefixes = [request["prefix_actions"] for request in requests]
        present_prefixes = [value for value in prefixes if value is not None]
        if present_prefixes:
            candidates = max(value.shape[0] for value in present_prefixes)
            length = max(value.shape[1] for value in present_prefixes)
            width = max(value.shape[2] for value in present_prefixes)
            prefix_array = np.full(
                (len(requests), candidates, length, width), np.nan, dtype=np.float32
            )
            for index, value in enumerate(prefixes):
                if value is not None:
                    prefix_array[
                        index,
                        : value.shape[0],
                        : value.shape[1],
                        : value.shape[2],
                    ] = value
            arrays["async_request_rtc_prefix_actions"] = prefix_array
        else:
            arrays["async_request_rtc_prefix_actions"] = np.empty((0, 0, 0, 0), dtype=np.float32)

        def generated_array(name: str, width: int) -> np.ndarray:
            values = [request[name] for request in requests]
            if not values:
                return np.empty((0, 0, 0, width), dtype=np.float32)
            expected = (self._num_candidates, self._policy_chunk_size, width)
            result = np.full((len(values), *expected), np.nan, dtype=np.float32)
            for index, value in enumerate(values):
                if value is None:
                    continue
                if value.shape != expected:
                    raise ValueError(f"request {name} does not match configured [N,H,D] dimensions")
                result[index] = value
            return result

        arrays["async_request_generated_policy_actions"] = generated_array(
            "generated_policy_actions", action_dim
        )
        arrays["async_request_generated_control_actions"] = generated_array(
            "generated_control_actions", action_dim
        )
        return arrays

    @staticmethod
    def _serialize_sparse(
        sparse: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        mapping = {
            "sparse_observation_episode_id": ("episode_id", np.int64),
            "sparse_observation_request_id": ("request_id", np.int64),
            "sparse_observation_generation": ("generation", np.int64),
            "sparse_observation_origin_action_count": ("origin_action_count", np.int64),
            "sparse_observation_activation_action_count": (
                "activation_action_count",
                np.int64,
            ),
            "sparse_observation_action_count": ("action_count", np.int64),
            "sparse_observation_event_role": ("event_role", np.str_),
        }
        arrays = {
            output: np.asarray([record[source] for record in sparse], dtype=dtype)
            for output, (source, dtype) in mapping.items()
        }
        keys = metadata["sparse_exact_observations"]["keys"]
        if sparse:
            expected = {
                key: (value.dtype, value.shape) for key, value in sparse[0]["values"].items()
            }
            for record in sparse[1:]:
                actual = {
                    key: (value.dtype, value.shape) for key, value in record["values"].items()
                }
                if actual != expected:
                    raise ValueError("sparse observation schema changed within the rollout")
        for item in keys:
            key = item["key"]
            arrays[item["array"]] = np.stack([record["values"][key] for record in sparse])
        return arrays


class DiagnosticActionProcessor:
    """Call one existing action processor once and observe its exact input/output."""

    def __init__(self, wrapped: Any, capture: RolloutCapture) -> None:
        self._wrapped = wrapped
        self._capture = capture

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def __call__(self, value: tuple[Any, Any]) -> Any:
        processor_input, _observation = value
        result = self._wrapped(value)
        self._capture.record_action_processor(processor_input, result)
        return result


class DiagnosticRobotWrapper:
    """Observe the existing robot wrapper without extra reads or dispatches."""

    def __init__(
        self,
        wrapped: Any,
        capture: RolloutCapture,
        *,
        dataset: Any | None = None,
    ) -> None:
        self._wrapped = wrapped
        self._capture = capture
        self._dataset = dataset

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def get_observation(self) -> Any:
        observation = self._wrapped.get_observation()
        self._capture.record_observation(observation)
        return observation

    def send_action(self, action: Any) -> Any:
        dataset_episode, dataset_frame = self._predict_dataset_frame_ref()
        result = self._wrapped.send_action(action)
        self._capture.record_command_sent(
            action,
            dataset_episode_index=dataset_episode,
            dataset_frame_index=dataset_frame,
        )
        return result

    def _predict_dataset_frame_ref(self) -> tuple[int | None, int | None]:
        writer = None if self._dataset is None else getattr(self._dataset, "writer", None)
        buffer = None if writer is None else getattr(writer, "episode_buffer", None)
        if buffer is None:
            return None, None
        try:
            return int(buffer["episode_index"]), int(buffer["size"])
        except (KeyError, TypeError, ValueError, AttributeError):
            logger.warning("Could not read dataset frame identity for diagnostic capture")
            return None, None


def install_diagnostic_capture(ctx: Any, capture: RolloutCapture) -> None:
    """Configure and install both transparent boundary wrappers on one context."""

    cfg = ctx.runtime.cfg
    inference = cfg.inference
    policy_config = ctx.policy.policy.config
    task = cfg.dataset.single_task if cfg.dataset is not None else cfg.task
    capture.configure(
        action_keys=ctx.data.ordered_action_keys,
        fps=float(cfg.fps),
        interpolation_multiplier=int(cfg.interpolation_multiplier),
        policy_chunk_size=int(policy_config.chunk_size),
        max_action_dim=int(policy_config.max_action_dim),
        num_candidates=int(inference.num_candidates),
        bank_refill_threshold=int(inference.bank_refill_threshold),
        guidance_horizon=int(inference.guidance_horizon),
        guidance_mode=str(inference.async_bank_guidance),
        selector_mode=str(inference.selector_mode),
        hysteresis_tau=inference.hysteresis_tau,
        phase0_seed=int(inference.phase0_seed),
        task=str(task),
        robot_type=str(ctx.hardware.robot_wrapper.robot_type),
    )
    ctx.hardware.robot_wrapper = DiagnosticRobotWrapper(
        ctx.hardware.robot_wrapper,
        capture,
        dataset=ctx.data.dataset,
    )
    ctx.processors.robot_action_processor = DiagnosticActionProcessor(
        ctx.processors.robot_action_processor,
        capture,
    )


__all__ = [
    "DiagnosticActionProcessor",
    "DiagnosticRobotWrapper",
    "RolloutCapture",
    "install_diagnostic_capture",
]
