from __future__ import annotations

import os
from copy import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import require_existing_directory

from dream_handoff.dataset import FINAL_DATASET_HF_REVISION, FINAL_DATASET_REPO_ID
from dream_handoff.inference import (
    DreamHandoffInferenceConfig,
    DreamHandoffInferenceEngine,
    SmolVLACandidateSampler,
)
from dream_handoff.inference.sampling import configure_rtc_prefix_guidance
from dream_handoff.r2dreamer import R2DreamerRuntime

pytestmark = pytest.mark.inference_artifacts


def _artifact(name: str) -> Path:
    value = os.environ.get(name)
    if value is None:
        pytest.skip(f"set {name} to run the real inference gate")
    path = Path(value)
    if not path.exists():
        pytest.fail(f"{name} does not exist: {path}")
    return path


def _raw_observation(sample: dict) -> dict[str, np.ndarray]:
    result = {"observation.state": sample["observation.state"].numpy()}
    for key in ("observation.images.context", "observation.images.wrist"):
        image = sample[key]
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"dataset image {key} is not CHW RGB")
        result[key] = image.permute(1, 2, 0).contiguous().numpy()
    return result


def test_real_smolvla_r2_chain_and_static_rtc_prefix(
    local_lerobot_dataset: None,
) -> None:
    """Offline only: load one dataset frame and never construct or connect a robot."""
    if not torch.cuda.is_available():
        pytest.skip("the recovered SmolVLA processor targets CUDA")
    policy_path = _artifact("DREAMHANDOFF_POLICY")
    checkpoint_path = _artifact("DREAMHANDOFF_R2_CHECKPOINT")
    dataset_path = require_existing_directory(
        str(_artifact("DREAMHANDOFF_DATASET")), "DREAMHANDOFF_DATASET"
    )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.utils import prepare_observation_for_inference

    dataset = LeRobotDataset(
        repo_id=FINAL_DATASET_REPO_ID,
        root=dataset_path,
        revision=FINAL_DATASET_HF_REVISION,
        video_backend=os.environ.get("DREAMHANDOFF_VIDEO_BACKEND", "pyav"),
        return_uint8=True,
    )
    sample = dataset[0]
    raw = _raw_observation(sample)
    task = sample["task"]

    policy = SmolVLAPolicy.from_pretrained(policy_path).to("cuda").eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(policy_path)
    )
    runtime = R2DreamerRuntime.from_checkpoint(checkpoint_path, "cuda")
    action_names = list(runtime.action_names)
    engine = DreamHandoffInferenceEngine(
        config=DreamHandoffInferenceConfig(
            r2_checkpoint=checkpoint_path,
            compile_world_model_imagination=False,
        ),
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        r2_runtime=runtime,
        dataset_features=dataset.meta.features,
        ordered_action_keys=action_names,
        task=task,
        fps=float(dataset.meta.fps),
        device="cuda",
        robot_type="so_follower",
    )
    engine.reset()
    engine.start()
    engine.resume()
    try:
        action = engine.get_action(raw)
        bank = engine.candidate_bank
        assert action.shape == (6,)
        assert bank.policy_actions.shape == (10, 35, 6)
        assert bank.control_actions.shape == (10, 35, 6)
        assert bank.dreamed_matching_features.shape == (10, 35, 2560)
        assert torch.isfinite(action).all()

        configure_rtc_prefix_guidance(policy)
        sampler = SmolVLACandidateSampler(policy, preprocessor, postprocessor, device="cuda")
        prepared = prepare_observation_for_inference(
            copy(raw), torch.device("cuda"), task, "so_follower"
        )
        prefix = bank.policy_actions[:, :15].detach().clone()
        guided_policy, guided_control = sampler.sample_candidate_bank(
            prepared,
            10,
            prefix_actions=prefix,
            inference_delay=7,
            execution_horizon=15,
        )
        assert guided_policy.shape == guided_control.shape == (10, 50, 6)
        assert torch.isfinite(guided_policy).all()
        assert torch.isfinite(guided_control).all()
    finally:
        engine.stop()
