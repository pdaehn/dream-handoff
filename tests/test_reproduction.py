from __future__ import annotations

import hashlib
import json
import logging
import os
import runpy
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dream_handoff.dataset import FINAL_DATASET_HF_REVISION, EpisodeSplit
from dream_handoff.r2dreamer.checkpoint import FINAL_R2_ARCHITECTURE_SIGNATURE
from dream_handoff.r2dreamer.training import final_model_config

ROOT = Path(__file__).parents[1]
R2_CONFIG = ROOT / "configs/r2/rectangle_s12_r64.json"
SMOLVLA_CONFIG = ROOT / "configs/smolvla/rectangle.json"
ARTIFACTS = ROOT / "configs/artifacts.json"
SPLIT = ROOT / "configs/data/episode_split.json"
LEROBOT_REVISION = "c841a0c25833b866970c5a31d66c2bff60f171de"
SMOLVLA_REVISION = "7e3d6e4c8ec1d43673b0ca96035e8995ed179601"
SMOLVLA_MODEL_SHA256 = "668cbfa273b30bd023ebcdc246e5fd0d533633bdf0d0d3c6ccd2891aef5dd64c"
JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def test_public_reproduction_path_deterministically_emits_five_jsons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/reproduce.py"))
    reproduce = namespace["reproduce"]
    globals_ = reproduce.__globals__

    monkeypatch.setitem(globals_, "_verify_file", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(globals_, "run_rollout", lambda *_args, **_kwargs: {})
    monkeypatch.setitem(globals_, "run_hysteresis", lambda *_args, **_kwargs: {})
    monkeypatch.setitem(globals_, "run_handoff_prediction", lambda *_args, **_kwargs: {})
    monkeypatch.setitem(globals_, "run_prefix_conditioning", lambda *_args, **_kwargs: {})
    monkeypatch.setitem(globals_, "curate_rollout", lambda *_args, **_kwargs: {"kind": "rollout"})
    monkeypatch.setitem(
        globals_, "curate_hysteresis", lambda *_args, **_kwargs: {"kind": "hysteresis"}
    )
    monkeypatch.setitem(globals_, "curate_handoff", lambda *_args, **_kwargs: {"kind": "handoff"})
    monkeypatch.setitem(globals_, "curate_prefix", lambda *_args, **_kwargs: {"kind": "prefix"})

    calibration_result = {"kind": "calibration", "deterministic": True}

    def fake_reconstruction(*, output_path: Path, **_kwargs: Any) -> dict[str, Any]:
        globals_["_write_json"](output_path, calibration_result)
        return calibration_result

    monkeypatch.setitem(globals_, "run_reconstruction", fake_reconstruction)

    common = {
        "diagnostic": tmp_path / "capture.npz",
        "calibration_dataset": tmp_path / "dataset",
        "calibration_policy": tmp_path / "policy",
        "r2_checkpoint": tmp_path / "latest.pt",
        "generated_banks": tmp_path / "generated.npz",
        "figures": None,
        "device": "cpu",
        "calibration_diagnostics": None,
        "calibration_reference_matrix": None,
        "policy_path": None,
    }
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    assert reproduce(SimpleNamespace(output_dir=run_a, **common)) == [
        "handoff_prediction.json",
        "hysteresis.json",
        "hysteresis_calibration.json",
        "prefix_conditioning.json",
        "rollout_characterization.json",
    ]
    assert reproduce(SimpleNamespace(output_dir=run_b, **common)) == [
        "handoff_prediction.json",
        "hysteresis.json",
        "hysteresis_calibration.json",
        "prefix_conditioning.json",
        "rollout_characterization.json",
    ]
    assert {path.name for path in run_a.glob("*.json")} == {
        "handoff_prediction.json",
        "hysteresis.json",
        "hysteresis_calibration.json",
        "prefix_conditioning.json",
        "rollout_characterization.json",
    }
    for first in sorted(run_a.glob("*.json")):
        assert first.read_bytes() == (run_b / first.name).read_bytes()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_public_r2_config_parses_and_matches_retained_trainer() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/train_r2.py"))
    settings = namespace["load_reproduction_config"](R2_CONFIG)
    config = _read_json(R2_CONFIG)
    model = final_model_config()

    assert config["artifact"] == "dreamhandoff-r2-rectangle-dynamics"
    for forbidden in ("/home/", "PycharmProjects", "wandb.ai", "api_key", "token"):
        assert forbidden.lower() not in R2_CONFIG.read_text().lower()
    assert config["architecture_signature"] == FINAL_R2_ARCHITECTURE_SIGNATURE
    assert config["model"] == json.loads(json.dumps(model.to_dict()))
    assert (model.stoch, model.discrete, model.deter) == (32, 16, 2048)
    assert model.stoch * model.discrete + model.deter == 2560
    assert (model.action_dim, model.image_channels, model.state_dim) == (6, 6, 6)
    assert settings == {
        "device": "cuda",
        "replay_storage_device": "cuda",
        "seed": 0,
        "steps": 25_000,
        "sequence_length": 64,
        "batch_size": 16,
        "num_workers": 4,
        "validation_interval": 500,
        "checkpoint_interval": 500,
        "validation_batches": 20,
        "learning_rate": 4e-5,
        "warmup_steps": 1_000,
        "cache_preprocessed_dataset": True,
    }


def test_smolvla_config_parses_with_pinned_upstream_lerobot(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch
    from lerobot.configs import parser as lerobot_parser
    from lerobot.configs import policies as policy_configs
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    config = _read_json(SMOLVLA_CONFIG)
    artifacts = _read_json(ARTIFACTS)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    split = EpisodeSplit.read(SPLIT)

    assert project["tool"]["uv"]["sources"]["lerobot"]["rev"] == LEROBOT_REVISION
    assert config["batch_size"] == 64
    assert config["steps"] == 29_200
    assert config["seed"] == 0
    assert config["save_freq"] == 2_000
    assert config["use_policy_training_preset"] is True
    assert config["rename_map"] == {
        "observation.images.context": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }
    assert "wandb" not in config
    assert not (ROOT / "scripts/train_smolvla.py").exists()

    base_config = tmp_path / "base-config.json"
    base_config.write_text(
        json.dumps(
            {
                "type": "smolvla",
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [6]},
                    "observation.images.camera1": {"type": "VISUAL", "shape": [3, 480, 640]},
                    "observation.images.camera2": {"type": "VISUAL", "shape": [3, 480, 640]},
                    "observation.images.camera3": {"type": "VISUAL", "shape": [3, 480, 640]},
                },
                "output_features": {"action": {"type": "ACTION", "shape": [6]}},
            }
        )
    )

    def local_base_config(*, repo_id: str, filename: str, **_kwargs: Any) -> str:
        assert repo_id == "lerobot/smolvla_base"
        assert filename == "config.json"
        return str(base_config)

    monkeypatch.setattr(policy_configs, "hf_hub_download", local_base_config)
    monkeypatch.setattr(policy_configs, "is_torch_device_available", lambda _device: True)

    # Keep the pinned training-interface gate independent of the publication reproduction guide.
    train_args = [
        "--config_path=configs/smolvla/rectangle.json",
        f"--dataset.repo_id={artifacts['dataset']['hf_repo']}",
        f"--dataset.root={tmp_path / 'dataset'}",
        f"--dataset.revision={FINAL_DATASET_HF_REVISION}",
        "--dataset.eval_split=0.2",
        "--dataset.episodes="
        + json.dumps(
            [*split.train_episode_indices, *split.validation_episode_indices], separators=(",", ":")
        ),
        f"--output_dir={tmp_path / 'policy-output'}",
        "--job_name=dreamhandoff-smolvla-rectangle-on-peg",
        "--batch_size=64",
        "--steps=29200",
        f"--policy.path={artifacts['smolvla']['pretrained_source']}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--policy.chunk_size=50",
        "--policy.n_action_steps=50",
        "--policy.num_steps=10",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--policy.train_state_proj=true",
        "--policy.use_peft=false",
        "--policy.compile_model=false",
        f"--policy.repo_id={artifacts['smolvla']['hf_repo']}",
        "--policy.push_to_hub=false",
        "--policy.private=true",
    ]

    captured: dict[str, TrainPipelineConfig] = {}

    def capture_config(cfg: TrainPipelineConfig) -> None:
        captured["config"] = cfg

    capture_config.__annotations__["cfg"] = TrainPipelineConfig
    monkeypatch.setattr(sys, "argv", ["lerobot-train", *train_args])
    lerobot_parser.wrap()(capture_config)()
    parsed = captured["config"]
    parsed.validate()

    assert parsed.dataset.repo_id == "pdaehn/dreamhandoff-so101-rectangle-on-peg"
    assert parsed.dataset.revision == FINAL_DATASET_HF_REVISION
    assert parsed.dataset.eval_split == 0.2
    assert parsed.dataset.episodes == [
        *split.train_episode_indices,
        *split.validation_episode_indices,
    ]
    assert parsed.steps == 29_200
    assert parsed.batch_size == 64
    assert parsed.seed == 0
    assert parsed.rename_map == config["rename_map"]

    assert isinstance(parsed.policy, SmolVLAConfig)
    assert parsed.policy.device == "cuda"
    assert parsed.policy.use_amp is False
    assert parsed.policy.chunk_size == 50
    assert parsed.policy.n_action_steps == 50
    assert parsed.policy.num_steps == 10
    assert parsed.policy.freeze_vision_encoder is True
    assert parsed.policy.train_expert_only is True
    assert parsed.policy.train_state_proj is True
    assert parsed.policy.use_peft is False
    assert parsed.policy.compile_model is False
    assert parsed.peft is None
    assert parsed.policy.robot_state_feature.shape == (6,)
    assert parsed.policy.action_feature.shape == (6,)
    assert list(parsed.policy.image_features) == [
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    ]
    assert artifacts["smolvla"]["features"]["state_names"] == JOINT_NAMES
    assert artifacts["smolvla"]["features"]["action_names"] == JOINT_NAMES

    assert isinstance(parsed.optimizer, AdamWConfig)
    assert parsed.optimizer.lr == 1e-4
    assert parsed.optimizer.betas == (0.9, 0.95)
    assert parsed.optimizer.eps == 1e-8
    assert parsed.optimizer.weight_decay == 1e-10
    assert parsed.optimizer.grad_clip_norm == 10
    assert isinstance(parsed.scheduler, CosineDecayWithWarmupSchedulerConfig)
    assert parsed.scheduler.num_warmup_steps == 1_000
    assert parsed.scheduler.num_decay_steps == 30_000
    assert parsed.scheduler.peak_lr == 1e-4
    assert parsed.scheduler.decay_lr == 2.5e-6

    optimizer = parsed.optimizer.build([torch.nn.Parameter(torch.zeros(()))])
    with caplog.at_level(logging.INFO):
        scheduler = parsed.scheduler.build(optimizer, parsed.steps)
    assert "Scaling warmup: 1000 → 973, decay: 30000 → 29200" in caplog.text
    assert scheduler.lr_lambdas[0](29_200) == pytest.approx(0.025)


def test_artifact_manifest_is_complete_portable_and_internally_consistent() -> None:
    manifest = _read_json(ARTIFACTS)
    rendered = ARTIFACTS.read_text()
    split = EpisodeSplit.read(SPLIT)

    assert "/home/" not in rendered
    assert manifest["dataset"]["episodes"] == 120
    assert manifest["dataset"]["canonical_split_sha256"] == (
        "cd0893cdbfd2d65198f8de8e55c8505b89bf40f998c3f846d50d4a42a03ffa21"
    )
    assert manifest["dataset"]["scientific_split_sha256"] == split.canonical_sha256
    assert manifest["dataset"]["episode_metadata_sha256"] == split.episode_metadata_sha256
    assert len(manifest["dataset"]["scientific_payload_files"]) == 8
    assert "README.md" not in manifest["dataset"]["scientific_payload_files"]
    expected_repos = {
        "dataset": "pdaehn/dreamhandoff-so101-rectangle-on-peg",
        "smolvla": "pdaehn/dreamhandoff-smolvla-rectangle-on-peg",
        "r2": "pdaehn/dreamhandoff-r2-rectangle-dynamics",
        "final_evidence": "pdaehn/dreamhandoff-so101-prospective-evaluation",
    }
    expected_revisions = {
        "dataset": FINAL_DATASET_HF_REVISION,
        "smolvla": SMOLVLA_REVISION,
        "r2": "b3510497b0ff135220ea92468d2efbd5908e4d41",
        "final_evidence": "4ff19fae23719daa0899c04166cb4d445fde0261",
    }
    for role, repo in expected_repos.items():
        artifact = manifest[role]
        assert artifact["hf_repo"] == repo
        assert artifact["hf_revision"] == expected_revisions[role]
        assert artifact["hf_revision"] != artifact.get("sha256")
        assert artifact["hf_revision"] != artifact.get("checkpoint_sha256")
    assert manifest["dataset"]["hf_revision"] == FINAL_DATASET_HF_REVISION
    assert manifest["smolvla"]["hf_revision"] == SMOLVLA_REVISION
    assert manifest["r2"]["name"] == "dreamhandoff-r2-rectangle-dynamics"
    assert manifest["final_evidence"]["diagnostic_name"] == "capture.npz"
    assert manifest["prefix_conditioning_replay"]["filename"] == "generated_banks.npz"
    assert manifest["prefix_conditioning_replay"]["sha256"] == (
        "3ce2ab0c8eb39b9650de98f3377f1b676e5131205f808dca16490c73982f3258"
    )
    assert manifest["prefix_conditioning_replay"]["hf_repo"] == expected_repos["final_evidence"]
    assert (
        manifest["prefix_conditioning_replay"]["hf_revision"]
        == manifest["final_evidence"]["hf_revision"]
    )
    assert manifest["smolvla"]["training"]["train_frames"] == 46_690
    assert manifest["smolvla"]["training"]["epochs"] == 40
    assert manifest["smolvla"]["training"]["steps"] == 29_200
    assert manifest["smolvla"]["hf_revision"] == SMOLVLA_REVISION
    assert manifest["smolvla"]["model_safetensors_sha256"] == SMOLVLA_MODEL_SHA256
    assert manifest["smolvla"]["lerobot_revision"] == LEROBOT_REVISION
    assert manifest["r2"]["checkpoint_filename"] == "latest.pt"
    assert manifest["r2"]["checkpoint_sha256"] == (
        "5260b7c5df88929e21bd60c8686fe7b9dc43db7b2c6238cf49524b42a33ac9fc"
    )
    assert manifest["r2"]["architecture_signature"] == FINAL_R2_ARCHITECTURE_SIGNATURE
    assert manifest["final_evidence"]["artifact_id"] == "prospective physical evaluation"
    assert manifest["final_evidence"]["sha256"] == (
        "c2c709ca5fbbaa673d8c31d71683fb5b63756596083e2b5f3d00f61975170bba"
    )
    assert manifest["final_evidence"]["physical_episodes"] == 24
    assert manifest["final_evidence"]["discarded_rerecord_capture_episode_ids"] == [4]


def test_prospective_configuration_serializes_controller() -> None:
    from dream_handoff.inference import FINAL_HYSTERESIS_TAU, DreamHandoffInferenceConfig

    config = _read_json(ROOT / "configs/prospective_evaluation.json")
    controller = dict(config["controller"])
    controller.pop("phase0_rng")
    controller["selector_mode"] = controller.pop("selector")
    parsed = DreamHandoffInferenceConfig(r2_checkpoint=Path("external/latest.pt"), **controller)
    serialized = json.loads(
        json.dumps(
            {
                **controller,
                "r2_checkpoint": str(parsed.r2_checkpoint),
            },
            sort_keys=True,
        )
    )

    assert parsed.hysteresis_tau == FINAL_HYSTERESIS_TAU == 0.06851652264595032
    assert serialized["hysteresis_tau"] == FINAL_HYSTERESIS_TAU
    assert config["artifact"]["capture_filename"] == "capture.npz"
    assert config["calibration"]["publication_result_sha256"] == _sha256(
        ROOT / config["calibration"]["result"]
    )
    assert config["observation_rename_map"] == {
        "observation.images.context": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }


def test_prospective_scientific_provenance() -> None:
    record = _read_json(ROOT / "configs/prospective_evaluation_provenance.json")
    assert record["capture_sha256"] == (
        "c2c709ca5fbbaa673d8c31d71683fb5b63756596083e2b5f3d00f61975170bba"
    )
    assert record["calibration_frozen_before_physical_evaluation"] is True
    assert record["collection_time_calibration_result_sha256"] == (
        "a839c729eb290633382c3118998bc72b47fe4a9d6b5338ac542440f7f84c175f"
    )
    assert record["selected_tau"] == 0.06851652264595032
    assert record["canonical_episode_ids"] == list(range(24))
    assert record["canonical_control_steps"] == 12257
    assert record["discarded_rerecord_capture_episode_ids"] == [4]


@pytest.mark.smolvla_artifact
def test_recovered_smolvla_metadata_matches_public_reproduction_config() -> None:
    configured = os.environ.get("DREAMHANDOFF_POLICY")
    if configured is None:
        pytest.skip("set DREAMHANDOFF_POLICY to run the real-policy metadata gate")
    policy_root = Path(configured)
    if not policy_root.is_dir():
        pytest.fail(f"DREAMHANDOFF_POLICY does not exist: {policy_root}")

    public = _read_json(SMOLVLA_CONFIG)
    train = _read_json(policy_root / "train_config.json")
    policy = _read_json(policy_root / "config.json")
    preprocessor = _read_json(policy_root / "policy_preprocessor.json")
    split = EpisodeSplit.read(SPLIT)

    assert train["dataset"]["episodes"] == [
        *split.train_episode_indices,
        *split.validation_episode_indices,
    ]
    assert train["dataset"]["eval_split"] == 0.2
    for key in ("seed", "num_workers", "batch_size", "steps", "save_freq"):
        assert train[key] == public[key]
    assert train["rename_map"] == public["rename_map"]
    assert train["policy"]["pretrained_path"] == "lerobot/smolvla_base"
    assert train["policy"]["device"] == "cuda"
    assert train["policy"]["use_amp"] is False
    expected_model_settings = {
        "chunk_size": 50,
        "n_action_steps": 50,
        "num_steps": 10,
        "freeze_vision_encoder": True,
        "train_expert_only": True,
        "train_state_proj": True,
        "use_peft": False,
        "compile_model": False,
    }
    for key, expected in expected_model_settings.items():
        assert train["policy"][key] == expected
    assert train["optimizer"] == {
        "type": "adamw",
        "lr": 0.0001,
        "weight_decay": 1e-10,
        "grad_clip_norm": 10.0,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
    }
    assert train["scheduler"] == {
        "type": "cosine_decay_with_warmup",
        "num_warmup_steps": 1000,
        "num_decay_steps": 30000,
        "peak_lr": 0.0001,
        "decay_lr": 2.5e-6,
    }
    assert policy["input_features"]["observation.state"]["shape"] == [6]
    assert list(policy["input_features"])[1:] == [
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    ]
    assert policy["output_features"]["action"]["shape"] == [6]
    assert preprocessor["steps"][0]["config"]["rename_map"] == public["rename_map"]
    normalized_features = preprocessor["steps"][5]["config"]["features"]
    assert normalized_features["observation.state"]["shape"] == [6]
    assert normalized_features["action"]["shape"] == [6]
    assert _sha256(policy_root / "model.safetensors") == SMOLVLA_MODEL_SHA256

    artifacts = _read_json(ARTIFACTS)
    assert artifacts["smolvla"]["model_settings"] == {
        "chunk_size": 50,
        "n_action_steps": 50,
        "flow_steps": 10,
        "freeze_vision_encoder": True,
        "train_expert_only": True,
        "train_state_proj": True,
        "use_peft": False,
        "use_amp": False,
        "compile_model": False,
    }
    assert artifacts["smolvla"]["features"]["state_names"] == JOINT_NAMES
    assert artifacts["smolvla"]["features"]["action_names"] == JOINT_NAMES
