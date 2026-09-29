from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dream_handoff.capture import (
    DiagnosticActionProcessor,
    DiagnosticRobotWrapper,
    RolloutCapture,
    load_diagnostic_capture,
)
from dream_handoff.inference import DreamHandoffInferenceConfig, DreamHandoffInferenceEngine
from dream_handoff.r2dreamer import R2State, R2Trajectory


class FakePolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            chunk_size=4,
            max_action_dim=4,
            use_amp=False,
            rtc_config=None,
        )
        self.calls = []
        self.reset_calls = 0

    def predict_action_chunk(self, batch, noise=None, **kwargs):
        self.calls.append((batch, noise.detach().clone(), kwargs))
        return noise[..., :2].clone()

    def reset(self) -> None:
        self.reset_calls += 1


class BlockingSecondPolicy(FakePolicy):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def predict_action_chunk(self, batch, noise=None, **kwargs):
        if len(self.calls) == 1:
            self.started.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("test did not release stale generation")
        return super().predict_action_chunk(batch, noise=noise, **kwargs)


class PolicyProcessor:
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
        self.observe_previous = []
        self.imagine_calls = []
        self.observe_count = 0

    def reset(self) -> None:
        self.observe_count = 0

    def compile_imagination(self, *_args) -> float:
        return 0.0

    def observe(self, _observation, previous_action) -> R2State:
        self.observe_count += 1
        self.observe_previous.append(
            None if previous_action is None else previous_action.detach().clone()
        )
        stoch = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
        deter = torch.tensor([[float(self.observe_count)]], dtype=torch.float32)
        return R2State(stoch, deter)

    def imagine_states(self, start: R2State, actions: torch.Tensor) -> R2Trajectory:
        self.imagine_calls.append(actions.detach().clone())
        candidates, horizon = actions.shape[:2]
        deter = actions[:, :, :1].clone()
        deter[:, 0] = start.deter.item()
        stoch = torch.zeros(candidates, horizon, 1, 2)
        stoch[..., 0] = 1.0
        return R2Trajectory(stoch, deter)

    @staticmethod
    def matching_feature(state: R2State) -> torch.Tensor:
        return state.deter

    @staticmethod
    def matching_features(trajectory: R2Trajectory) -> torch.Tensor:
        return trajectory.deter


class RobotActionProcessor:
    def __init__(self) -> None:
        self.calls = []
        self.reset_calls = 0

    def __call__(self, value):
        action, observation = value
        self.calls.append((action, observation))
        return {key: scalar + 0.125 for key, scalar in action.items()}

    def reset(self) -> None:
        self.reset_calls += 1


class FakeRobot:
    robot_type = "fake_robot"

    def __init__(self) -> None:
        self.observation_calls = 0
        self.commands = []

    def get_observation(self):
        self.observation_calls += 1
        value = float(self.observation_calls)
        return {
            "joint_a": value,
            "joint_b": -value,
            "camera": np.full((2, 2, 3), self.observation_calls, dtype=np.uint8),
        }

    def send_action(self, action):
        self.commands.append(dict(action))
        return action


def _run_sequence(tmp_path: Path, *, capture_enabled: bool) -> dict:
    torch.manual_seed(1234)
    policy = FakePolicy()
    preprocessor = PolicyProcessor()
    postprocessor = PolicyProcessor(offset=0.5)
    r2 = FakeR2()
    capture = RolloutCapture(tmp_path / "parity.npz") if capture_enabled else None
    if capture is not None:
        capture.configure(
            action_keys=["joint_a", "joint_b"],
            fps=30.0,
            interpolation_multiplier=1,
            policy_chunk_size=4,
            max_action_dim=4,
            num_candidates=3,
            bank_refill_threshold=1,
            guidance_horizon=1,
            guidance_mode="plain",
            selector_mode="absolute_hysteresis",
            hysteresis_tau=0.1,
            phase0_seed=0,
            task="synthetic task",
            robot_type="fake_robot",
        )
    config = DreamHandoffInferenceConfig(
        r2_checkpoint=Path("external.pt"),
        num_candidates=3,
        max_bank_age=4,
        bank_refill_threshold=1,
        guidance_horizon=1,
        compile_world_model_imagination=False,
        selector_mode="absolute_hysteresis",
        hysteresis_tau=0.1,
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
        ordered_action_keys=["joint_a", "joint_b"],
        task="synthetic task",
        fps=30.0,
        device="cpu",
        robot_type="fake_robot",
        event_sink=None if capture is None else capture.event_sink,
    )
    processor_inner = RobotActionProcessor()
    robot_inner = FakeRobot()
    processor = (
        processor_inner if capture is None else DiagnosticActionProcessor(processor_inner, capture)
    )
    robot = robot_inner if capture is None else DiagnosticRobotWrapper(robot_inner, capture)
    assert isinstance(processor, DiagnosticActionProcessor) is capture_enabled
    assert isinstance(robot, DiagnosticRobotWrapper) is capture_enabled

    engine.reset()
    engine.start()
    engine.resume()
    served = []
    selections = []
    try:
        for step in range(6):
            raw = robot.get_observation()
            obs_frame = {
                "observation.images.context": raw["camera"],
                "observation.state": np.asarray([raw["joint_a"], raw["joint_b"]], dtype=np.float32),
            }
            action = engine.get_action(obs_frame)
            served.append(action.clone())
            selections.append(engine.last_selection.index)
            action_dict = {"joint_a": action[0].item(), "joint_b": action[1].item()}
            processed = processor((action_dict, raw))
            robot.send_action(processed)
            if step == 3:
                deadline = time.monotonic() + 2.0
                while not engine.has_ready_bank:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("asynchronous candidate bank was not ready")
                    time.sleep(0.001)
    finally:
        engine.stop()

    capture_summary = None
    if capture is not None:
        path = capture.save()
        with load_diagnostic_capture(path) as loaded:
            capture_summary = {
                "steps": loaded.control_step_count,
                "requests": loaded.request_count,
                "banks": loaded.candidate_bank_count,
                "handoffs": len(loaded.handoffs()),
            }
    return {
        "selections": selections,
        "served": torch.stack(served),
        "commands": robot_inner.commands,
        "policy_noise": [call[1] for call in policy.calls],
        "policy_call_count": len(policy.calls),
        "request_count": len(policy.calls),
        "preprocessor_calls": preprocessor.calls,
        "postprocessor_calls": postprocessor.calls,
        "r2_observe_previous": r2.observe_previous,
        "r2_imagine_calls": r2.imagine_calls,
        "activation_count": len(r2.imagine_calls),
        "rng_state": torch.get_rng_state(),
        "robot_observation_calls": robot_inner.observation_calls,
        "capture_summary": capture_summary,
    }


def test_capture_off_and_on_have_identical_controller_behavior(tmp_path: Path) -> None:
    without_capture = _run_sequence(tmp_path / "off", capture_enabled=False)
    with_capture = _run_sequence(tmp_path / "on", capture_enabled=True)

    assert without_capture["selections"] == with_capture["selections"]
    torch.testing.assert_close(without_capture["served"], with_capture["served"])
    assert without_capture["commands"] == with_capture["commands"]
    assert without_capture["policy_call_count"] == with_capture["policy_call_count"] == 2
    assert without_capture["request_count"] == with_capture["request_count"] == 2
    assert without_capture["activation_count"] == with_capture["activation_count"] == 2
    assert without_capture["preprocessor_calls"] == with_capture["preprocessor_calls"]
    assert without_capture["postprocessor_calls"] == with_capture["postprocessor_calls"]
    assert len(without_capture["r2_observe_previous"]) == len(with_capture["r2_observe_previous"])
    for expected, actual in zip(
        without_capture["r2_observe_previous"],
        with_capture["r2_observe_previous"],
        strict=True,
    ):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(expected, actual)
    for expected, actual in zip(
        without_capture["r2_imagine_calls"],
        with_capture["r2_imagine_calls"],
        strict=True,
    ):
        torch.testing.assert_close(expected, actual)
    for expected, actual in zip(
        without_capture["policy_noise"], with_capture["policy_noise"], strict=True
    ):
        torch.testing.assert_close(expected, actual)
    torch.testing.assert_close(without_capture["rng_state"], with_capture["rng_state"])
    assert without_capture["robot_observation_calls"] == 6
    assert with_capture["robot_observation_calls"] == 6
    assert without_capture["capture_summary"] is None
    assert with_capture["capture_summary"] == {
        "steps": 6,
        "requests": 2,
        "banks": 2,
        "handoffs": 1,
    }


def test_reset_preserves_completed_stale_payload_in_mixed_capture(tmp_path: Path) -> None:
    torch.manual_seed(1234)
    capture = RolloutCapture(tmp_path / "stale-completed.npz")
    capture.configure(
        action_keys=["joint_a", "joint_b"],
        fps=30.0,
        interpolation_multiplier=1,
        policy_chunk_size=4,
        max_action_dim=4,
        num_candidates=3,
        bank_refill_threshold=1,
        guidance_horizon=1,
        guidance_mode="plain",
        selector_mode="absolute_hysteresis",
        hysteresis_tau=0.1,
        phase0_seed=0,
        task="synthetic task",
        robot_type="fake_robot",
    )
    policy = BlockingSecondPolicy()
    engine = DreamHandoffInferenceEngine(
        config=DreamHandoffInferenceConfig(
            r2_checkpoint=Path("external.pt"),
            num_candidates=3,
            max_bank_age=4,
            bank_refill_threshold=1,
            guidance_horizon=1,
            compile_world_model_imagination=False,
            selector_mode="absolute_hysteresis",
            hysteresis_tau=0.1,
            phase0_seed=0,
        ),
        policy=policy,
        preprocessor=PolicyProcessor(),
        postprocessor=PolicyProcessor(offset=0.5),
        r2_runtime=FakeR2(),
        dataset_features={
            "action": {"names": ["joint_a", "joint_b"]},
            "observation.state": {},
        },
        ordered_action_keys=["joint_a", "joint_b"],
        task="synthetic task",
        fps=30.0,
        device="cpu",
        robot_type="fake_robot",
        event_sink=capture.event_sink,
    )
    engine.reset()
    engine.start()
    engine.resume()
    reset_thread: threading.Thread | None = None
    try:
        for step in range(4):
            engine.get_action({"observation.state": np.asarray([step, -step], dtype=np.float32)})
        assert policy.started.wait(timeout=2)
        reset_thread = threading.Thread(target=engine.reset)
        reset_thread.start()
        deadline = time.monotonic() + 2.0
        while engine._epoch < 2:  # noqa: SLF001
            if time.monotonic() >= deadline:
                raise TimeoutError("reset did not invalidate the in-flight generation")
            time.sleep(0.001)
        policy.release.set()
        reset_thread.join(timeout=2)
        assert not reset_thread.is_alive()
        assert engine.candidate_bank is None
    finally:
        policy.release.set()
        if reset_thread is not None:
            reset_thread.join(timeout=2)
        engine.stop()

    with load_diagnostic_capture(capture.save()) as loaded:
        requests = loaded.requests()
        assert len(requests) == 2
        assert requests[0].activated and requests[0].generation_completed
        assert requests[1].stale and not requests[1].activated
        assert requests[1].generation_completed
        assert "completed after reset" in requests[1].stale_reason
        assert np.isfinite(loaded.array("async_request_generation_noise")).all()
        generated_policy = loaded.array("async_request_generated_policy_actions")
        generated_control = loaded.array("async_request_generated_control_actions")
        assert generated_policy.shape == generated_control.shape == (2, 3, 4, 2)
        assert np.isfinite(generated_policy).all()
        assert np.isfinite(generated_control).all()
        loaded.require_exact_replay()
