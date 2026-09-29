from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dream_handoff.inference import DreamHandoffInferenceConfig, DreamHandoffInferenceEngine
from dream_handoff.r2dreamer import R2State, R2Trajectory

FIXTURE = Path(__file__).parent / "fixtures" / "wp3_frozen_lifecycle.json"


class FakePolicy:
    def __init__(
        self,
        chunk_size: int,
        *,
        block_from_call: int | None = None,
        fail_on_call: int | None = None,
    ) -> None:
        self.config = SimpleNamespace(
            chunk_size=chunk_size,
            max_action_dim=4,
            use_amp=False,
            rtc_config=None,
        )
        self.rtc_processor = None
        self.block_from_call = block_from_call
        self.fail_on_call = fail_on_call
        self.calls: list[dict] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.reset_calls = 0

    def init_rtc_processor(self) -> None:
        self.rtc_processor = object() if self.config.rtc_config is not None else None

    def predict_action_chunk(self, batch, noise=None, **kwargs):
        call = len(self.calls) + 1
        self.calls.append({"batch": batch, "noise": noise, "kwargs": kwargs})
        if self.fail_on_call == call:
            raise ValueError("planned generation failure")
        if self.block_from_call is not None and call >= self.block_from_call:
            self.started.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("test did not release policy generation")
        candidates = noise.shape[0]
        phase = torch.arange(self.config.chunk_size, device=noise.device).view(1, -1, 1)
        candidate = torch.arange(candidates, device=noise.device).view(-1, 1, 1) * 100
        return (call * 1000 + candidate + phase).expand(-1, -1, 2).float().clone()

    def reset(self) -> None:
        self.reset_calls += 1


class Processor:
    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset
        self.calls = 0
        self.reset_calls = 0

    def __call__(self, value):
        self.calls += 1
        return value + self.offset if torch.is_tensor(value) else value

    def reset(self) -> None:
        self.reset_calls += 1


class FakeR2:
    action_names = ("joint_a", "joint_b")
    observation_keys = ("observation.state",)

    def __init__(self) -> None:
        self.counter = 0
        self.observe_previous: list[torch.Tensor | None] = []
        self.imagine_calls: list[tuple[float, torch.Tensor]] = []
        self.reset_calls = 0
        self.compile_calls: list[tuple[int, int]] = []

    def reset(self) -> None:
        self.reset_calls += 1
        self.counter = 0

    def compile_imagination(self, num_candidates: int, horizon: int) -> float:
        self.compile_calls.append((num_candidates, horizon))
        return 0.0

    def observe(self, observation, previous_action) -> R2State:  # noqa: ARG002
        self.counter += 1
        self.observe_previous.append(
            None if previous_action is None else previous_action.detach().clone()
        )
        value = torch.tensor([[float(self.counter)]])
        stoch = torch.nn.functional.one_hot(torch.zeros(1, 1, dtype=torch.long), 2).float()
        return R2State(stoch, value)

    def imagine_states(self, start_state: R2State, actions: torch.Tensor) -> R2Trajectory:
        start = float(start_state.deter.item())
        self.imagine_calls.append((start, actions.detach().clone()))
        candidates, horizon = actions.shape[:2]
        deter = actions[:, :, :1].clone()
        deter[:, 0] = start
        categories = torch.zeros(candidates, horizon, 1, dtype=torch.long, device=actions.device)
        stoch = torch.nn.functional.one_hot(categories, 2).float()
        return R2Trajectory(stoch, deter)

    @staticmethod
    def matching_feature(state: R2State) -> torch.Tensor:
        return state.deter

    @staticmethod
    def matching_features(trajectory: R2Trajectory) -> torch.Tensor:
        return trajectory.deter


class StepClock:
    def __init__(self, step: float) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


def observation(value: float) -> dict[str, np.ndarray]:
    return {"observation.state": np.asarray([value, -value], dtype=np.float32)}


def make_engine(
    *,
    num_candidates: int = 2,
    chunk_size: int = 8,
    max_bank_age: int = 5,
    refill: int = 2,
    guidance_horizon: int = 2,
    guidance: str = "plain",
    selector: str = "absolute_hysteresis",
    compile_imagination: bool = False,
    block_from_call: int | None = None,
    fail_on_call: int | None = None,
    fps: float = 10.0,
    clock=None,
    device: str = "cpu",
):
    policy = FakePolicy(
        chunk_size,
        block_from_call=block_from_call,
        fail_on_call=fail_on_call,
    )
    preprocessor = Processor()
    postprocessor = Processor(0.5)
    r2 = FakeR2()
    events: list[tuple[str, dict]] = []
    config = DreamHandoffInferenceConfig(
        r2_checkpoint=Path("external.pt"),
        num_candidates=num_candidates,
        max_bank_age=max_bank_age,
        bank_refill_threshold=refill,
        async_bank_guidance=guidance,
        guidance_horizon=guidance_horizon,
        compile_world_model_imagination=compile_imagination,
        selector_mode=selector,
        hysteresis_tau=(0.25 if selector == "absolute_hysteresis" else None),
        phase0_seed=0,
    )
    engine = DreamHandoffInferenceEngine(
        config=config,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        r2_runtime=r2,
        dataset_features={
            "action": {"names": ["joint_a", "joint_b"]},
            "observation.state": {},
        },
        ordered_action_keys=["joint_b", "joint_a"],
        task="test task",
        fps=fps,
        device=device,
        robot_type="test_robot",
        event_sink=lambda name, values: events.append((name, dict(values))),
        monotonic_clock=clock or time.perf_counter,
    )
    engine.reset()
    engine.start()
    engine.resume()
    return engine, policy, preprocessor, postprocessor, r2, events


def wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("condition was not reached")
        time.sleep(0.001)


def serve(engine: DreamHandoffInferenceEngine, start: int, stop: int) -> list[torch.Tensor]:
    return [engine.get_action(observation(float(tick))) for tick in range(start, stop)]


def test_initial_bank_is_plain_blocking_common_activation_and_returns_ordered_action() -> None:
    engine, policy, _, _, r2, events = make_engine()
    try:
        exact_observation = observation(0.0)
        returned = engine.get_action(exact_observation)
        bank = engine.candidate_bank
        assert bank is not None
        assert bank.origin_action_count == 0
        assert bank.horizon == 5
        assert len(r2.observe_previous) == len(r2.imagine_calls) == 1
        assert policy.calls[0]["kwargs"] == {}
        request = next(values for name, values in events if name == "bank_requested")
        ready = next(values for name, values in events if name == "generation_ready")
        activation = next(values for name, values in events if name == "bank_activated")
        served = next(values for name, values in events if name == "action_served")
        assert request["guidance_mode"] == "plain"
        assert request["prefix_actions"] is None
        assert request["request_observation"] is exact_observation
        assert activation["activation_observation"] is exact_observation
        assert activation["actual_elapsed"] == 0
        assert "generated" not in ready
        assert "bank" not in activation
        assert "selection" not in served
        assert "live_state" not in served
        torch.testing.assert_close(ready["sampled_noise"], policy.calls[0]["noise"])
        selected = engine.last_selection.index
        canonical = bank.control_actions[selected, 0]
        torch.testing.assert_close(engine.previous_action, canonical)
        torch.testing.assert_close(returned, canonical.flip(0).cpu())
    finally:
        policy.release.set()
        engine.stop()


def test_previous_served_control_action_is_observed_on_the_next_tick_exactly_once() -> None:
    engine, policy, _, _, r2, _ = make_engine()
    try:
        engine.get_action(observation(0.0))
        first = engine.previous_action
        engine.get_action(observation(1.0))
        assert len(r2.observe_previous) == 2
        assert r2.observe_previous[0] is None
        torch.testing.assert_close(r2.observe_previous[1], first)
    finally:
        policy.release.set()
        engine.stop()


def test_refill_schedule_actual_crop_and_fresh_activation_posterior() -> None:
    engine, policy, _, _, r2, events = make_engine(block_from_call=2)
    try:
        serve(engine, 0, 4)
        assert policy.started.wait(timeout=2)
        request = [v for n, v in events if n == "bank_requested"][-1]
        assert (request["request_action_count"], request["request_phase"]) == (3, 3)
        assert len(r2.imagine_calls) == 1
        policy.release.set()
        wait_until(lambda: engine.has_ready_bank)
        engine.get_action(observation(4.0))
        bank = engine.candidate_bank
        assert bank is not None
        assert (bank.origin_action_count, bank.horizon) == (4, 5)
        assert len(r2.imagine_calls) == 2
        assert r2.imagine_calls[-1][0] == 5.0
        full_generated = 2000 + torch.arange(8).view(1, -1, 1)
        expected_first = full_generated[0, 1, 0] + 0.5
        assert bank.control_actions[0, 0, 0] == expected_first
        activation = [v for n, v in events if n == "bank_activated"][-1]
        assert activation["actual_elapsed"] == 1
        assert engine.last_selection.rule == "seeded_uniform"
    finally:
        policy.release.set()
        engine.stop()


def test_static_rtc_warmup_and_real_request_use_same_index_policy_prefix() -> None:
    clock = StepClock(0.25)
    engine, policy, _, _, _, events = make_engine(
        guidance="static_rtc",
        block_from_call=3,
        clock=clock,
    )
    try:
        engine.get_action(observation(0.0))
        initial_bank = engine.candidate_bank
        assert initial_bank is not None
        assert len(policy.calls) == 2
        assert policy.calls[0]["kwargs"] == {}
        torch.testing.assert_close(
            policy.calls[1]["kwargs"]["prev_chunk_left_over"],
            initial_bank.policy_actions[:, :2],
        )
        assert policy.calls[1]["kwargs"]["inference_delay"] == 0
        serve(engine, 1, 4)
        assert policy.started.wait(timeout=2)
        request = [v for n, v in events if n == "bank_requested"][-1]
        torch.testing.assert_close(request["prefix_actions"], initial_bank.policy_actions[:, 3:5])
        assert request["predicted_delay"] == math.ceil(0.25 * 10.0) == 3
        policy.release.set()
        wait_until(lambda: engine.has_ready_bank)
        kwargs = policy.calls[2]["kwargs"]
        assert kwargs["inference_delay"] == 3
        assert kwargs["execution_horizon"] == 2
        assert kwargs["prev_chunk_left_over"].shape == (2, 2, 2)
    finally:
        policy.release.set()
        engine.stop()


def test_reset_rejects_old_epoch_and_preserves_selector_rng_and_latency() -> None:
    engine, policy, _, _, r2, events = make_engine(
        num_candidates=10,
        block_from_call=2,
    )
    try:
        engine.get_action(observation(0.0))
        first = engine.last_selection.index
        serve(engine, 1, 4)
        assert policy.started.wait(timeout=2)
        reset_thread = threading.Thread(target=engine.reset)
        reset_thread.start()
        wait_until(lambda: engine._epoch >= 2)  # noqa: SLF001
        policy.release.set()
        reset_thread.join(timeout=2)
        assert not reset_thread.is_alive()
        assert engine.candidate_bank is None
        assert engine.actions_served == 0
        assert r2.reset_calls == 2
        assert any(name == "stale_generation" for name, _ in events)
        assert len(engine._latency_tracker) == 1  # noqa: SLF001
        engine.get_action(observation(10.0))
        assert (first, engine.last_selection.index) == (4, 9)
    finally:
        policy.release.set()
        engine.stop()


def test_pause_resume_gates_pending_work_and_stop_clears_bank() -> None:
    engine, policy, _, _, _, _ = make_engine(block_from_call=2)
    try:
        engine.pause()
        serve(engine, 0, 4)
        time.sleep(0.06)
        assert len(policy.calls) == 1
        assert engine.request_in_flight
        engine.resume()
        assert policy.started.wait(timeout=2)
    finally:
        policy.release.set()
        engine.stop()
    assert engine.candidate_bank is None


def test_worker_failure_propagates_and_sets_shutdown_event() -> None:
    engine, policy, _, _, _, events = make_engine(fail_on_call=2)
    try:
        serve(engine, 0, 4)
        wait_until(lambda: engine.failed)
        with pytest.raises(RuntimeError, match="bank worker failed") as error:
            engine.get_action(observation(4.0))
        assert isinstance(error.value.__cause__, ValueError)
        assert any(name == "worker_failure" for name, _ in events)
    finally:
        policy.release.set()
        engine.stop()


def test_fail_closed_when_active_bank_expires() -> None:
    engine, policy, _, _, _, _ = make_engine(block_from_call=2)
    try:
        serve(engine, 0, 5)
        assert policy.started.wait(timeout=2)
        with pytest.raises(RuntimeError, match="expired"):
            engine.get_action(observation(5.0))
    finally:
        policy.release.set()
        engine.stop()


def test_fail_closed_when_ready_generation_is_too_late_to_crop() -> None:
    engine, policy, _, _, _, _ = make_engine(chunk_size=6, block_from_call=2)
    try:
        serve(engine, 0, 4)
        assert policy.started.wait(timeout=2)
        with engine._state_lock:  # noqa: SLF001
            engine._actions_served = 10  # noqa: SLF001
        policy.release.set()
        wait_until(lambda: engine.has_ready_bank)
        with pytest.raises(RuntimeError, match="too late"):
            engine.get_action(observation(10.0))
    finally:
        policy.release.set()
        engine.stop()


def test_start_controls_optional_fixed_shape_r2_compile() -> None:
    engine, policy, _, _, r2, _ = make_engine(compile_imagination=True)
    try:
        assert r2.compile_calls == [(2, 5)]
        engine.start()
        assert r2.compile_calls == [(2, 5)]
    finally:
        policy.release.set()
        engine.stop()


def test_frozen_final_lifecycle_fixture_parity() -> None:
    fixture = json.loads(FIXTURE.read_text())
    engine, policy, _, _, _, events = make_engine(
        num_candidates=10,
        chunk_size=50,
        max_bank_age=35,
        refill=15,
        guidance_horizon=15,
        block_from_call=2,
    )
    try:
        serve(engine, 0, 21)
        assert policy.started.wait(timeout=2)
        request = [v for n, v in events if n == "bank_requested"][-1]
        assert (
            request["request_action_count"] == fixture["request_schedule"]["request_action_count"]
        )
        assert request["request_phase"] == fixture["request_schedule"]["request_phase"]
        serve(engine, 21, 24)
        policy.release.set()
        wait_until(lambda: engine.has_ready_bank)
        engine.get_action(observation(24.0))
        bank = engine.candidate_bank
        activation = [v for n, v in events if n == "bank_activated"][-1]
        assert activation["actual_elapsed"] == fixture["activation"]["actual_elapsed"]
        assert bank.horizon == fixture["activation"]["usable_horizon"]
        assert bank.origin_action_count == fixture["activation"]["activation_origin"]
        phase0 = [
            values["selected_candidate_index"]
            for name, values in events
            if name == "action_served" and values["phase"] == 0
        ]
        assert phase0 == fixture["phase0_seed_0_n10"][:2]
    finally:
        policy.release.set()
        engine.stop()


def test_frozen_final_static_rtc_fixture_parity() -> None:
    fixture = json.loads(FIXTURE.read_text())["static_rtc"]
    clock = StepClock(0.7)
    engine, policy, _, _, _, events = make_engine(
        num_candidates=10,
        chunk_size=50,
        max_bank_age=35,
        refill=15,
        guidance_horizon=15,
        guidance="static_rtc",
        block_from_call=3,
        clock=clock,
    )
    try:
        serve(engine, 0, 21)
        assert policy.started.wait(timeout=2)
        request = [v for n, v in events if n == "bank_requested"][-1]
        start, stop = fixture["prefix_slice"]
        active = engine.candidate_bank
        torch.testing.assert_close(request["prefix_actions"], active.policy_actions[:, start:stop])
        assert request["request_phase"] == fixture["request_phase"]
        assert request["predicted_delay"] == fixture["predicted_delay"]
        policy.release.set()
        wait_until(lambda: engine.has_ready_bank)
        kwargs = policy.calls[2]["kwargs"]
        assert kwargs["execution_horizon"] == fixture["execution_horizon"]
        assert kwargs["prev_chunk_left_over"].shape[:2] == (10, 15)
    finally:
        policy.release.set()
        engine.stop()


@pytest.mark.cuda
def test_cuda_worker_stream_event_and_publication_path() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    engine, policy, _, _, _, _ = make_engine(device="cuda", block_from_call=2)
    try:
        serve(engine, 0, 4)
        assert policy.started.wait(timeout=2)
        policy.release.set()
        wait_until(lambda: engine.has_ready_bank)
        engine.get_action(observation(4.0))
        assert engine.candidate_bank.policy_actions.device.type == "cuda"
        assert engine.candidate_bank.origin_action_count == 4
    finally:
        policy.release.set()
        engine.stop()


class _ObservedPublicationEvent(torch.cuda.Event):
    """CUDA event that snapshots request input when a worker could consume it."""

    def __new__(cls, **_kwargs):
        return super().__new__(cls, enable_timing=True)

    def __init__(
        self,
        *,
        destination: torch.Tensor,
        observed: torch.Tensor,
        worker_stream: torch.cuda.Stream,
        event_factory,
    ) -> None:
        self.destination = destination
        self.observed = observed
        self.worker_stream = worker_stream
        self.consumer_done = event_factory(enable_timing=True)
        self.was_recorded = False

    def record(self, stream=None) -> None:
        super().record(stream)
        self.was_recorded = True
        with torch.cuda.stream(self.worker_stream):
            self.worker_stream.wait_event(self)
            self.observed.copy_(self.destination)
            self.consumer_done.record()


class _DelayedCudaPrefix:
    """Queue a delayed finite prefix copy when the request snapshots this value."""

    def __init__(
        self,
        expected: torch.Tensor,
        destination: torch.Tensor,
        ready_event: torch.cuda.Event,
        publication: dict[str, _ObservedPublicationEvent],
    ) -> None:
        self.expected = expected
        self.destination = destination
        self.ready_event = ready_event
        self.publication = publication

    def __getitem__(self, item):
        return _DelayedCudaPrefix(
            self.expected[item],
            self.destination[item],
            self.ready_event,
            self.publication,
        )

    def detach(self):
        return self

    def clone(self) -> torch.Tensor:
        publication = self.publication.get("event")
        if publication is not None and publication.was_recorded:
            torch.cuda.current_stream().wait_event(publication.consumer_done)
        torch.cuda._sleep(100_000_000)  # noqa: SLF001
        self.destination.add_(self.expected)
        self.ready_event.record()
        return self.destination


@pytest.mark.cuda
def test_static_rtc_cuda_request_publication_waits_for_delayed_prefix_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device("cuda")
    engine, policy, _, _, _, _ = make_engine(
        device="cuda",
        guidance="static_rtc",
        chunk_size=20,
        max_bank_age=20,
        refill=15,
        guidance_horizon=15,
    )
    try:
        expected = torch.full((2, 20, 2), 2222.0, device=device)
        destination = torch.zeros_like(expected)
        observed = torch.zeros_like(expected[:, :15])
        torch.cuda.synchronize(device)
        event_factory = torch.cuda.Event
        prefix_ready = event_factory(enable_timing=True)
        worker_stream = torch.cuda.Stream(device=device, priority=-1)
        publication: dict[str, _ObservedPublicationEvent] = {}

        def publication_event(*_args, **_kwargs):
            event = _ObservedPublicationEvent(
                destination=destination[:, :15],
                observed=observed,
                worker_stream=worker_stream,
                event_factory=event_factory,
            )
            publication["event"] = event
            return event

        monkeypatch.setattr(torch.cuda, "Event", publication_event)
        old_bank = SimpleNamespace(
            policy_actions=_DelayedCudaPrefix(
                expected,
                destination,
                prefix_ready,
                publication,
            ),
            origin_action_count=0,
            horizon=20,
        )
        producer_stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(producer_stream):
            request = engine._make_request(  # noqa: SLF001
                {"observation.state": torch.zeros(1, 2, device=device)},
                old_bank=old_bank,
            )

        request.cuda_input_ready.synchronize()
        request.cuda_input_ready.consumer_done.synchronize()
        prefix_ready.synchronize()
        prefix_to_request_ms = prefix_ready.elapsed_time(request.cuda_input_ready)
        torch.testing.assert_close(observed, expected[:, :15])
        assert prefix_to_request_ms >= 0.0, (
            "request readiness was recorded before the Static RTC prefix copy completed; "
            f"prefix-to-readiness delta was {prefix_to_request_ms:.3f} ms"
        )
        torch.testing.assert_close(request.prefix_actions, expected[:, :15])
    finally:
        policy.release.set()
        engine.stop()
        torch.cuda.synchronize(device)
