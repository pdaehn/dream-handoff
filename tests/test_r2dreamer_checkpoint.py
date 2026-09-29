from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dream_handoff.r2dreamer import (
    R2DreamerActionNormalizer,
    R2DreamerModelConfig,
    R2DreamerPreprocessingConfig,
    R2DreamerRuntime,
    load_r2dreamer_checkpoint,
)
from dream_handoff.r2dreamer.checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_FORMAT_VERSION,
    R2_UPSTREAM_COMMIT,
)
from dream_handoff.r2dreamer.config import architecture_signature
from dream_handoff.r2dreamer.model import R2DreamerWorldModel


def _payload() -> dict:
    config = R2DreamerModelConfig(
        action_dim=2,
        image_channels=0,
        state_dim=3,
        image_size=(16, 16),
        stoch=4,
        deter=16,
        hidden=8,
        discrete=4,
        blocks=4,
        encoder_depth=2,
        encoder_mults=(1, 2),
        vector_layers=1,
        vector_units=8,
    )
    preprocessing = R2DreamerPreprocessingConfig(
        image_keys=(), image_size=(16, 16), state_key="observation.state"
    )
    normalization = R2DreamerActionNormalizer(
        action_names=("joint_a", "joint_b"), q01=(0.0, 10.0), q99=(10.0, 50.0)
    )
    return {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": R2DreamerWorldModel(config).state_dict(),
        "model_config": config.to_dict(),
        "preprocessing_config": preprocessing.to_dict(),
        "action_normalization": normalization.to_dict(),
        "step": 12,
        "r2_upstream_commit": R2_UPSTREAM_COMMIT,
        "metadata": {
            "architecture_signature": architecture_signature(config),
            "action_metadata": {"names": list(normalization.action_names)},
            "action_normalization": normalization.to_dict(),
        },
    }


def _save_latest(tmp_path: Path, payload: dict | None = None) -> Path:
    path = tmp_path / "latest.pt"
    torch.save(_payload() if payload is None else payload, path)
    return path


def test_checkpoint_loads_strict_model_config_and_runtime(tmp_path: Path) -> None:
    path = _save_latest(tmp_path)

    loaded = load_r2dreamer_checkpoint(path)
    runtime = R2DreamerRuntime.from_checkpoint(path, "cpu")

    assert loaded.step == 12
    assert loaded.model.config.action_dim == 2
    assert loaded.model.config.stoch == 4
    assert loaded.model.config.discrete == 4
    assert loaded.model.config.deter == 16
    assert runtime.action_names == ("joint_a", "joint_b")
    assert runtime.observation_keys == ("observation.state",)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda payload: payload.pop("action_normalization"), "missing required keys"),
        (lambda payload: payload.update(format_version=99), "unsupported R2 checkpoint"),
        (
            lambda payload: payload["metadata"].update(architecture_signature="wrong"),
            "architecture signature is inconsistent",
        ),
        (
            lambda payload: payload["model_state_dict"].pop(
                next(iter(payload["model_state_dict"]))
            ),
            "model weights are incompatible",
        ),
    ],
)
def test_incompatible_checkpoint_fails_clearly(tmp_path: Path, change, message: str) -> None:
    payload = _payload()
    change(payload)
    path = _save_latest(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        load_r2dreamer_checkpoint(path)


def test_runtime_accepts_compatible_checkpoint_with_arbitrary_basename(tmp_path: Path) -> None:
    path = tmp_path / "new-wp2b-checkpoint.pt"
    torch.save(_payload(), path)

    runtime = R2DreamerRuntime.from_checkpoint(path, "cpu")

    assert runtime.model.config.action_dim == 2
    assert runtime.action_names == ("joint_a", "joint_b")
