from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from lerobot.rollout import SyncInferenceConfig, SyncInferenceEngine

from dream_handoff import lerobot_compat
from dream_handoff.inference import FINAL_HYSTERESIS_TAU, DreamHandoffInferenceConfig
from dream_handoff.r2dreamer import R2DreamerRuntime

_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "dreamhandoff_rollout_script", Path(__file__).parents[1] / "scripts" / "rollout.py"
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
rollout_script = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(rollout_script)

ORDER_FIXTURE = Path(__file__).parent / "fixtures" / "wp4_frozen_rollout_order.json"
ACTION_NAMES = ("shoulder_pan.pos", "shoulder_lift.pos")
OBSERVATION_KEYS = (
    "observation.images.context",
    "observation.images.wrist",
    "observation.state",
)


class FakeR2Runtime:
    action_names = ACTION_NAMES
    observation_keys = OBSERVATION_KEYS
    events: list[str] | None = None
    failure: BaseException | None = None
    loaded: FakeR2Runtime | None = None

    @classmethod
    def from_checkpoint(cls, path, device):
        if cls.events is not None:
            cls.events.append("validate_r2")
        if cls.failure is not None:
            raise cls.failure
        cls.loaded = cls()
        cls.loaded.path = path
        cls.loaded.device = device
        return cls.loaded


class FakeDreamHandoffEngine:
    created: list[dict] = []
    events: list[str] | None = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.start_calls = 0
        self.stop_calls = 0
        type(self).created.append(kwargs)
        if type(self).events is not None:
            type(self).events.append("inject_engine")

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


class FakeConnectedDevice:
    def __init__(self, *, disconnect_error: BaseException | None = None) -> None:
        self.is_connected = True
        self.disconnect_error = disconnect_error
        self.disconnect_calls = 0
        self.send_action_calls = 0

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        if self.disconnect_error is not None:
            raise self.disconnect_error
        self.is_connected = False

    def send_action(self, _action) -> None:
        self.send_action_calls += 1


@pytest.fixture(autouse=True)
def block_real_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every compatibility test must explicitly install a hardware-free context builder."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a test attempted to call LeRobot's real hardware context builder")

    monkeypatch.setattr(lerobot_compat, "build_rollout_context", forbidden)
    monkeypatch.setattr(lerobot_compat, "R2DreamerRuntime", FakeR2Runtime)
    monkeypatch.setattr(lerobot_compat, "DreamHandoffInferenceEngine", FakeDreamHandoffEngine)
    FakeR2Runtime.events = None
    FakeR2Runtime.failure = None
    FakeR2Runtime.loaded = None
    FakeDreamHandoffEngine.created = []
    FakeDreamHandoffEngine.events = None


def _config(tmp_path: Path) -> SimpleNamespace:
    inference = DreamHandoffInferenceConfig(r2_checkpoint=tmp_path / "r2.pt")
    policy_config = SimpleNamespace(
        input_features={
            "observation.images.camera1": SimpleNamespace(shape=(3, 256, 256)),
            "observation.images.camera2": SimpleNamespace(shape=(3, 256, 256)),
            "observation.state": SimpleNamespace(shape=(2,)),
        },
        output_features={"action": SimpleNamespace(shape=(2,))},
        action_feature_names=None,
    )
    return SimpleNamespace(
        inference=inference,
        policy=policy_config,
        strategy=object(),
        rename_map={
            "observation.images.context": "observation.images.camera1",
            "observation.images.wrist": "observation.images.camera2",
        },
        dataset=None,
        task="place the rectangle on the pegs",
        device="cpu",
        fps=30.0,
        use_torch_compile=False,
    )


def _install_fake_context_builder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    events: list[str] | None = None,
    placeholder_factory=None,
    robot: FakeConnectedDevice | None = None,
    teleop: FakeConnectedDevice | None = None,
):
    calls: list[tuple[object, object, object]] = []
    owned: dict[str, object] = {}
    robot = robot or FakeConnectedDevice()

    def build(placeholder_cfg, shutdown_event):
        calls.append((placeholder_cfg, placeholder_cfg.inference, shutdown_event))
        if events is not None:
            events.append("build_context")
        policy = SimpleNamespace(config=SimpleNamespace(chunk_size=50, max_action_dim=32))
        preprocessor = SimpleNamespace()
        postprocessor = SimpleNamespace()
        dataset_features = {
            "action": {"names": list(ACTION_NAMES)},
            **{key: {} for key in OBSERVATION_KEYS},
        }
        ordered_action_keys = list(ACTION_NAMES)
        if placeholder_factory is None:
            placeholder = SyncInferenceEngine(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset_features=dataset_features,
                ordered_action_keys=ordered_action_keys,
                task=placeholder_cfg.task,
                device=placeholder_cfg.device,
                robot_type="fake_robot",
            )
        else:
            placeholder = placeholder_factory()
        context = SimpleNamespace(
            runtime=SimpleNamespace(cfg=placeholder_cfg, shutdown_event=shutdown_event),
            hardware=SimpleNamespace(
                robot_wrapper=SimpleNamespace(robot_type="fake_robot", inner=robot),
                teleop=teleop,
            ),
            policy=SimpleNamespace(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                inference=placeholder,
            ),
            data=SimpleNamespace(
                dataset_features=dataset_features,
                ordered_action_keys=ordered_action_keys,
            ),
        )
        owned.update(
            context=context,
            placeholder=placeholder,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            robot=robot,
            teleop=teleop,
        )
        return context

    monkeypatch.setattr(lerobot_compat, "build_rollout_context", build)
    return calls, owned


def test_r2_failure_prevents_hardware_context_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    context_calls, _ = _install_fake_context_builder(monkeypatch)
    FakeR2Runtime.failure = ValueError("invalid R2 checkpoint")

    with pytest.raises(ValueError, match="invalid R2 checkpoint"):
        lerobot_compat.build_dreamhandoff_rollout_context(cfg, Event())

    assert context_calls == []


def test_r2_policy_metadata_mismatch_fails_before_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    cfg.policy.action_feature_names = ["wrong_a", "wrong_b"]
    context_calls, _ = _install_fake_context_builder(monkeypatch)

    with pytest.raises(ValueError, match="canonical action names"):
        lerobot_compat.build_dreamhandoff_rollout_context(cfg, Event())

    assert context_calls == []


def test_one_shallow_context_copy_is_injected_and_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    real_inference = cfg.inference
    shutdown_event = Event()
    context_calls, owned = _install_fake_context_builder(monkeypatch)

    ctx = lerobot_compat.build_dreamhandoff_rollout_context(cfg, shutdown_event)

    assert len(context_calls) == 1
    placeholder_cfg, construction_inference, received_shutdown = context_calls[0]
    assert placeholder_cfg is not cfg
    assert placeholder_cfg.policy is cfg.policy
    assert isinstance(construction_inference, SyncInferenceConfig)
    assert cfg.inference is real_inference
    assert received_shutdown is shutdown_event

    engine = ctx.policy.inference
    assert isinstance(engine, FakeDreamHandoffEngine)
    assert engine is not owned["placeholder"]
    assert ctx.runtime.cfg.inference is real_inference
    kwargs = engine.kwargs
    assert kwargs["config"] is real_inference
    assert kwargs["r2_runtime"] is FakeR2Runtime.loaded
    assert kwargs["policy"] is owned["policy"]
    assert kwargs["preprocessor"] is owned["preprocessor"]
    assert kwargs["postprocessor"] is owned["postprocessor"]
    assert kwargs["dataset_features"] is owned["dataset_features"]
    assert kwargs["ordered_action_keys"] is owned["ordered_action_keys"]
    assert kwargs["task"] == cfg.task
    assert kwargs["fps"] == cfg.fps
    assert kwargs["device"] == cfg.device
    assert kwargs["robot_type"] == "fake_robot"
    assert kwargs["shutdown_event"] is shutdown_event
    assert engine.start_calls == engine.stop_calls == 0
    assert owned["robot"].disconnect_calls == 0


def test_placeholder_must_be_exact_stock_sync_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    robot = FakeConnectedDevice()
    teleop = FakeConnectedDevice()
    _install_fake_context_builder(
        monkeypatch,
        placeholder_factory=object,
        robot=robot,
        teleop=teleop,
    )

    with pytest.raises(TypeError, match="stock SyncInferenceEngine"):
        lerobot_compat.build_dreamhandoff_rollout_context(cfg, Event())

    assert FakeDreamHandoffEngine.created == []
    assert robot.disconnect_calls == teleop.disconnect_calls == 1
    assert robot.send_action_calls == teleop.send_action_calls == 0


def test_engine_construction_failure_disconnects_without_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    robot = FakeConnectedDevice()
    teleop = FakeConnectedDevice()
    _install_fake_context_builder(monkeypatch, robot=robot, teleop=teleop)
    injection_error = RuntimeError("engine construction failed")

    class FailingEngine:
        def __init__(self, **_kwargs):
            raise injection_error

    monkeypatch.setattr(lerobot_compat, "DreamHandoffInferenceEngine", FailingEngine)

    with pytest.raises(RuntimeError) as caught:
        lerobot_compat.build_dreamhandoff_rollout_context(cfg, Event())

    assert caught.value is injection_error
    assert robot.disconnect_calls == teleop.disconnect_calls == 1
    assert robot.send_action_calls == teleop.send_action_calls == 0


def test_keyboard_interrupt_during_injection_disconnects_without_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    robot = FakeConnectedDevice()
    teleop = FakeConnectedDevice()
    _install_fake_context_builder(monkeypatch, robot=robot, teleop=teleop)
    injection_error = KeyboardInterrupt("operator interrupted injection")

    class InterruptingEngine:
        def __init__(self, **_kwargs):
            raise injection_error

    monkeypatch.setattr(lerobot_compat, "DreamHandoffInferenceEngine", InterruptingEngine)

    with pytest.raises(KeyboardInterrupt) as caught:
        lerobot_compat.build_dreamhandoff_rollout_context(cfg, Event())

    assert caught.value is injection_error
    assert robot.disconnect_calls == teleop.disconnect_calls == 1
    assert robot.send_action_calls == teleop.send_action_calls == 0


def test_cleanup_failure_does_not_mask_injection_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _config(tmp_path)
    cleanup_error = RuntimeError("disconnect failed")
    robot = FakeConnectedDevice(disconnect_error=cleanup_error)
    teleop = FakeConnectedDevice()
    _install_fake_context_builder(monkeypatch, robot=robot, teleop=teleop)
    injection_error = RuntimeError("engine construction failed")

    class FailingEngine:
        def __init__(self, **_kwargs):
            raise injection_error

    monkeypatch.setattr(lerobot_compat, "DreamHandoffInferenceEngine", FailingEngine)

    with pytest.raises(RuntimeError) as caught:
        lerobot_compat.build_dreamhandoff_rollout_context(cfg, Event())

    assert caught.value is injection_error
    assert robot.disconnect_calls == 1
    assert teleop.disconnect_calls == 1
    assert robot.send_action_calls == teleop.send_action_calls == 0
    assert "Could not disconnect hardware after inference injection failed" in caplog.text


def test_strategy_construction_failure_without_capture_disconnects_every_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    robot = FakeConnectedDevice()
    teleop = FakeConnectedDevice()
    context = SimpleNamespace(
        hardware=SimpleNamespace(
            robot_wrapper=SimpleNamespace(inner=robot),
            teleop=teleop,
        )
    )
    strategy_error = RuntimeError("strategy construction failed")
    monkeypatch.setattr(
        rollout_script,
        "build_dreamhandoff_rollout_context",
        lambda *_args, **_kwargs: context,
    )

    def fail_strategy(_config):
        raise strategy_error

    monkeypatch.setattr(rollout_script, "create_strategy", fail_strategy)

    with pytest.raises(RuntimeError) as caught:
        rollout_script.run_rollout(cfg, Event())

    assert caught.value is strategy_error
    assert robot.disconnect_calls == teleop.disconnect_calls == 1
    assert robot.send_action_calls == teleop.send_action_calls == 0


def test_policy_torch_compile_is_rejected_before_context_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    cfg.use_torch_compile = True
    context_calls: list[object] = []
    monkeypatch.setattr(
        rollout_script,
        "build_dreamhandoff_rollout_context",
        lambda *_args, **_kwargs: context_calls.append(object()),
    )

    with pytest.raises(ValueError, match="Policy torch compilation is unsupported"):
        rollout_script.run_rollout(cfg, Event())

    assert context_calls == []


def test_context_builder_failure_does_not_attempt_context_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    build_error = RuntimeError("context build failed")
    cleanup_calls: list[object] = []

    def fail_build(*_args, **_kwargs):
        raise build_error

    monkeypatch.setattr(lerobot_compat, "build_rollout_context", fail_build)
    monkeypatch.setattr(
        lerobot_compat,
        "_disconnect_failed_context",
        lambda ctx: cleanup_calls.append(ctx),
    )

    with pytest.raises(RuntimeError) as caught:
        lerobot_compat.build_dreamhandoff_rollout_context(cfg, Event())

    assert caught.value is build_error
    assert cleanup_calls == []


def test_final_inference_configuration_is_preserved(tmp_path: Path) -> None:
    config = _config(tmp_path).inference

    assert config.num_candidates == 10
    assert config.max_bank_age == 35
    assert config.bank_refill_threshold == 15
    assert config.guidance_horizon == 15
    assert config.compile_world_model_imagination is True
    assert config.selector_mode == "absolute_hysteresis"
    assert config.hysteresis_tau == FINAL_HYSTERESIS_TAU
    assert config.phase0_seed == 0
    assert config.async_bank_guidance == "plain"


def test_frozen_order_and_stock_strategy_own_engine_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    events: list[str] = []
    FakeR2Runtime.events = events
    FakeDreamHandoffEngine.events = events
    monkeypatch.setattr(
        SyncInferenceEngine,
        "start",
        lambda _self: pytest.fail("the construction-only sync placeholder was started"),
    )
    _install_fake_context_builder(monkeypatch, events=events)

    def create_strategy(strategy_config):
        assert strategy_config is cfg.strategy
        # Observing the real config here pins restoration before strategy creation.
        assert ctx_holder["ctx"].runtime.cfg.inference is cfg.inference
        events.extend(["restore_config", "create_strategy"])
        return strategy

    class Strategy:
        def setup(self, ctx):
            events.append("setup")
            assert ctx.policy.inference is engine_holder["engine"]
            ctx.policy.inference.start()

        def run(self, ctx):
            events.append("run")

        def teardown(self, ctx):
            events.append("teardown")
            ctx.policy.inference.stop()

    strategy = Strategy()
    ctx_holder: dict[str, object] = {}
    engine_holder: dict[str, object] = {}
    real_builder = lerobot_compat.build_dreamhandoff_rollout_context

    def build_and_remember(config, shutdown_event):
        ctx = real_builder(config, shutdown_event)
        ctx_holder["ctx"] = ctx
        engine_holder["engine"] = ctx.policy.inference
        assert ctx.policy.inference.start_calls == ctx.policy.inference.stop_calls == 0
        return ctx

    monkeypatch.setattr(rollout_script, "build_dreamhandoff_rollout_context", build_and_remember)
    monkeypatch.setattr(rollout_script, "create_strategy", create_strategy)

    ctx = rollout_script.run_rollout(cfg, Event())

    assert events == json.loads(ORDER_FIXTURE.read_text())
    assert ctx.policy.inference.start_calls == 1
    assert ctx.policy.inference.stop_calls == 1


@pytest.mark.parametrize("failure_stage", ["setup", "run"])
def test_stock_teardown_runs_when_setup_or_run_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    cfg = _config(tmp_path)
    _, owned = _install_fake_context_builder(monkeypatch)
    ctx = lerobot_compat.build_dreamhandoff_rollout_context(cfg, Event())
    calls: list[str] = []

    class FailingStrategy:
        def setup(self, received):
            calls.append("setup")
            received.policy.inference.start()
            if failure_stage == "setup":
                raise RuntimeError("setup failed")

        def run(self, _received):
            calls.append("run")
            raise RuntimeError("run failed")

        def teardown(self, received):
            calls.append("teardown")
            received.policy.inference.stop()

    monkeypatch.setattr(rollout_script, "build_dreamhandoff_rollout_context", lambda *_args: ctx)
    monkeypatch.setattr(rollout_script, "create_strategy", lambda _config: FailingStrategy())

    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        rollout_script.run_rollout(cfg, Event())

    expected = ["setup", "teardown"] if failure_stage == "setup" else ["setup", "run", "teardown"]
    assert calls == expected
    assert ctx.policy.inference.start_calls == 1
    assert ctx.policy.inference.stop_calls == 1
    assert owned["placeholder"] is not ctx.policy.inference


def test_launcher_help_is_hardware_free() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/rollout.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--inference.type" in result.stdout
    assert "dream_handoff" in result.stdout
    assert "--capture" in result.stdout


def test_capture_installs_before_strategy_and_saves_after_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    cfg.capture = tmp_path / "capture.npz"
    events: list[str] = []
    context = SimpleNamespace()

    class FakeCapture:
        def __init__(self, output_path):
            self.output_path = output_path
            self.event_sink = object()

        def record_rollout_failure(self, _error):
            events.append("capture_failure")

        def save(self):
            events.append("capture_save")
            return self.output_path

    capture_holder = {}

    def make_capture(path):
        capture = FakeCapture(path)
        capture_holder["capture"] = capture
        return capture

    def build(_cfg, _shutdown, *, event_sink):
        events.append("build")
        assert event_sink is capture_holder["capture"].event_sink
        return context

    class Strategy:
        def setup(self, received):
            assert received is context
            events.append("setup")

        def run(self, received):
            assert received is context
            events.append("run")

        def teardown(self, received):
            assert received is context
            events.append("teardown")

    monkeypatch.setattr(rollout_script, "RolloutCapture", make_capture)
    monkeypatch.setattr(rollout_script, "build_dreamhandoff_rollout_context", build)
    monkeypatch.setattr(
        rollout_script,
        "install_diagnostic_capture",
        lambda received, capture: events.append("install"),
    )
    monkeypatch.setattr(rollout_script, "create_strategy", lambda _cfg: Strategy())

    assert rollout_script.run_rollout(cfg, Event()) is context
    assert events == ["build", "install", "setup", "run", "teardown", "capture_save"]


def test_pre_context_failure_skips_unconfigured_capture_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    cfg.capture = tmp_path / "capture.npz"
    build_error = RuntimeError("context build failed")

    def fail_build(*_args, **_kwargs):
        raise build_error

    monkeypatch.setattr(rollout_script, "build_dreamhandoff_rollout_context", fail_build)

    with pytest.raises(RuntimeError) as caught:
        rollout_script.run_rollout(cfg, Event())

    assert caught.value is build_error
    assert not cfg.capture.exists()


def test_configured_capture_is_saved_after_later_rollout_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    cfg.capture = tmp_path / "capture.npz"
    rollout_error = RuntimeError("rollout failed")
    events: list[str] = []

    class FakeCapture:
        def __init__(self, output_path):
            self.output_path = output_path
            self.event_sink = object()

        def record_rollout_failure(self, _error):
            events.append("capture_failure")

        def save(self):
            events.append("capture_save")
            self.output_path.write_bytes(b"diagnostic")
            return self.output_path

    class Strategy:
        def setup(self, _ctx):
            events.append("setup")

        def run(self, _ctx):
            events.append("run")
            raise rollout_error

        def teardown(self, _ctx):
            events.append("teardown")

    monkeypatch.setattr(rollout_script, "RolloutCapture", FakeCapture)
    monkeypatch.setattr(
        rollout_script,
        "build_dreamhandoff_rollout_context",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(rollout_script, "install_diagnostic_capture", lambda *_args: None)
    monkeypatch.setattr(rollout_script, "create_strategy", lambda _cfg: Strategy())

    with pytest.raises(RuntimeError) as caught:
        rollout_script.run_rollout(cfg, Event())

    assert caught.value is rollout_error
    assert events == [
        "setup",
        "run",
        "capture_failure",
        "teardown",
        "capture_failure",
        "capture_save",
    ]
    assert cfg.capture.read_bytes() == b"diagnostic"


def test_capture_save_failure_does_not_mask_rollout_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _config(tmp_path)
    cfg.capture = tmp_path / "capture.npz"
    rollout_error = RuntimeError("rollout failed")

    class FakeCapture:
        output_path = cfg.capture
        event_sink = object()

        def __init__(self, _path):
            pass

        def record_rollout_failure(self, _error):
            pass

        def save(self):
            raise OSError("capture save failed")

    class Strategy:
        def setup(self, _ctx):
            pass

        def run(self, _ctx):
            raise rollout_error

        def teardown(self, _ctx):
            pass

    monkeypatch.setattr(rollout_script, "RolloutCapture", FakeCapture)
    monkeypatch.setattr(
        rollout_script,
        "build_dreamhandoff_rollout_context",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(rollout_script, "install_diagnostic_capture", lambda *_args: None)
    monkeypatch.setattr(rollout_script, "create_strategy", lambda _cfg: Strategy())

    with pytest.raises(RuntimeError) as caught:
        rollout_script.run_rollout(cfg, Event())

    assert caught.value is rollout_error
    assert "preserving rollout failure" in caplog.text


def test_capture_save_failure_after_normal_rollout_is_propagated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    cfg.capture = tmp_path / "capture.npz"
    save_error = OSError("capture save failed")

    class FakeCapture:
        output_path = cfg.capture
        event_sink = object()

        def __init__(self, _path):
            pass

        def record_rollout_failure(self, _error):
            pass

        def save(self):
            raise save_error

    class Strategy:
        def setup(self, _ctx):
            pass

        def run(self, _ctx):
            pass

        def teardown(self, _ctx):
            pass

    monkeypatch.setattr(rollout_script, "RolloutCapture", FakeCapture)
    monkeypatch.setattr(
        rollout_script,
        "build_dreamhandoff_rollout_context",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(rollout_script, "install_diagnostic_capture", lambda *_args: None)
    monkeypatch.setattr(rollout_script, "create_strategy", lambda _cfg: Strategy())

    with pytest.raises(OSError) as caught:
        rollout_script.run_rollout(cfg, Event())

    assert caught.value is save_error


def test_capture_install_failure_disconnects_context_without_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    cfg.capture = tmp_path / "capture.npz"
    robot = FakeConnectedDevice()
    context = SimpleNamespace(
        hardware=SimpleNamespace(
            robot_wrapper=SimpleNamespace(inner=robot),
            teleop=None,
        )
    )
    install_error = RuntimeError("capture install failed")

    class FakeCapture:
        output_path = cfg.capture
        event_sink = object()

        def __init__(self, _path):
            pass

        def record_rollout_failure(self, _error):
            pass

        def save(self):
            raise RuntimeError("not configured")

    monkeypatch.setattr(rollout_script, "RolloutCapture", FakeCapture)
    monkeypatch.setattr(
        rollout_script,
        "build_dreamhandoff_rollout_context",
        lambda *_args, **_kwargs: context,
    )

    def fail_install(*_args):
        raise install_error

    monkeypatch.setattr(rollout_script, "install_diagnostic_capture", fail_install)

    with pytest.raises(RuntimeError) as caught:
        rollout_script.run_rollout(cfg, Event())

    assert caught.value is install_error
    assert robot.disconnect_calls == 1
    assert robot.send_action_calls == 0


@pytest.mark.r2_checkpoint
def test_real_r2_loads_before_patched_hardware_free_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = os.environ.get("DREAMHANDOFF_R2_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("set DREAMHANDOFF_R2_CHECKPOINT to run the real WP4 compatibility gate")

    cfg = _config(tmp_path)
    cfg.inference.r2_checkpoint = Path(checkpoint)
    cfg.policy.output_features["action"].shape = (6,)
    context_calls, _ = _install_fake_context_builder(monkeypatch)
    monkeypatch.setattr(lerobot_compat, "R2DreamerRuntime", R2DreamerRuntime)

    ctx = lerobot_compat.build_dreamhandoff_rollout_context(cfg, Event())

    assert len(context_calls) == 1
    assert isinstance(ctx.policy.inference.kwargs["r2_runtime"], R2DreamerRuntime)
    assert ctx.policy.inference.kwargs["r2_runtime"].action_names
