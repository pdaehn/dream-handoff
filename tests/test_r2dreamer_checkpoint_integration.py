from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from dream_handoff.r2dreamer import (
    FINAL_R2_ARCHITECTURE_SIGNATURE,
    FINAL_R2_CHECKPOINT_SHA256,
    R2DreamerRuntime,
    checkpoint_sha256,
    load_r2dreamer_checkpoint,
)
from dream_handoff.r2dreamer.config import architecture_signature

pytestmark = pytest.mark.r2_checkpoint
ROOT = Path(__file__).parents[1]


def _external_checkpoint() -> Path:
    configured = os.environ.get("DREAMHANDOFF_R2_CHECKPOINT")
    if configured is None:
        pytest.skip("set DREAMHANDOFF_R2_CHECKPOINT to run the real-checkpoint gate")
    path = Path(configured)
    if not path.is_file():
        pytest.fail(f"DREAMHANDOFF_R2_CHECKPOINT does not exist: {path}")
    return path


def test_final_checkpoint_identity_loading_and_frozen_runtime_fixture() -> None:
    path = _external_checkpoint()
    assert path.name == "latest.pt"
    assert checkpoint_sha256(path) == FINAL_R2_CHECKPOINT_SHA256

    loaded = load_r2dreamer_checkpoint(path, device="cpu")
    config = loaded.model.config
    public_config = json.loads((ROOT / "configs/r2/rectangle_s12_r64.json").read_text())
    assert loaded.step == 25_000
    assert architecture_signature(config) == FINAL_R2_ARCHITECTURE_SIGNATURE
    assert public_config["architecture_signature"] == architecture_signature(config)
    assert public_config["model"] == json.loads(json.dumps(config.to_dict()))
    assert public_config["preprocessing"] == json.loads(
        json.dumps(loaded.preprocessor.config.to_dict())
    )
    assert sum(parameter.numel() for parameter in loaded.model.parameters()) == 9_678_944
    assert (config.action_dim, config.image_channels, config.state_dim) == (6, 6, 6)
    assert (config.stoch, config.discrete, config.deter) == (32, 16, 2048)
    assert config.image_size == (64, 64)
    assert loaded.action_normalizer.action_names == (
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    )
    assert loaded.preprocessor.config.image_keys == (
        "observation.images.context",
        "observation.images.wrist",
    )
    assert loaded.preprocessor.config.state_key == "observation.state"

    runtime = R2DreamerRuntime.from_checkpoint(path, "cpu")
    observation = {
        "observation.images.context": torch.zeros(3, 64, 64),
        "observation.images.wrist": torch.ones(3, 64, 64),
        "observation.state": torch.arange(6, dtype=torch.float32),
    }
    state = runtime.observe(observation, previous_action=None)
    actions = torch.stack((torch.zeros(3, 6), torch.ones(3, 6)), dim=0)
    trajectory = runtime.imagine_states(state, actions)

    expected_deter = torch.tensor(
        [
            -0.020858274772763252,
            0.000376444892026484,
            0.052761003375053406,
            0.11940033733844757,
            0.05998414754867554,
            -0.03566216677427292,
            0.006217617075890303,
            0.023278657346963882,
        ]
    )
    expected_trajectory_deter = torch.tensor(
        [
            [
                [-0.020858274772763252, 0.000376444892026484, 0.052761003375053406],
                [-0.2098078578710556, -0.004650147631764412, 0.03949834406375885],
                [-0.2345745861530304, 0.04620738700032234, 0.05612519383430481],
            ],
            [
                [-0.020858274772763252, 0.000376444892026484, 0.052761003375053406],
                [-0.20504438877105713, -0.005258042830973864, 0.039980411529541016],
                [-0.23188062012195587, 0.04321325570344925, 0.058568380773067474],
            ],
        ]
    )
    torch.testing.assert_close(state.deter[0, :8], expected_deter, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        trajectory.deter[:, :, :3], expected_trajectory_deter, rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        runtime.matching_features(trajectory)[:, 0],
        runtime.matching_feature(state).expand(2, -1),
        rtol=0,
        atol=0,
    )

    recurrent = runtime.observe(
        observation,
        previous_action=torch.tensor([0.0, -100.0, 0.0, 70.0, 0.0, 10.0]),
    )
    expected_recurrent_deter = torch.tensor(
        [
            -0.2577005624771118,
            0.005327139049768448,
            0.01957963965833187,
            0.1029818058013916,
            0.02479500137269497,
            0.10161733627319336,
            0.10860437899827957,
            -0.02435273677110672,
        ]
    )
    expected_categories = torch.tensor(
        [
            9,
            11,
            8,
            2,
            7,
            13,
            4,
            15,
            15,
            8,
            4,
            12,
            6,
            6,
            1,
            4,
            15,
            9,
            7,
            11,
            13,
            8,
            5,
            3,
            6,
            10,
            13,
            5,
            15,
            0,
            6,
            1,
        ]
    )
    torch.testing.assert_close(
        recurrent.deter[0, :8], expected_recurrent_deter, rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(torch.argmax(recurrent.stoch[0], dim=-1), expected_categories)
