from __future__ import annotations

import os
from pathlib import Path

import pytest
from conftest import require_existing_directory

from dream_handoff.r2dreamer import R2DreamerRuntime
from dream_handoff.r2dreamer.config import R2DreamerTrainingConfig
from dream_handoff.r2dreamer.data import read_final_episode_split
from dream_handoff.r2dreamer.training import train_world_model


def _external_paths() -> tuple[Path, Path]:
    dataset = os.environ.get("DREAMHANDOFF_DATASET")
    split = os.environ.get("DREAMHANDOFF_SPLIT")
    if not dataset or not split:
        pytest.skip("set DREAMHANDOFF_DATASET and DREAMHANDOFF_SPLIT for the real-data gate")
    dataset_path = require_existing_directory(dataset, "DREAMHANDOFF_DATASET")
    split_path = Path(split)
    if not split_path.is_file():
        pytest.fail(f"DREAMHANDOFF_SPLIT does not exist: {split_path}")
    return dataset_path, split_path


@pytest.mark.r2_training
def test_real_dataset_provenance_and_checkpoint_runtime_smoke(
    tmp_path: Path, local_lerobot_dataset: None
) -> None:
    dataset, split_path = _external_paths()
    split = read_final_episode_split(split_path)
    assert len(split.train_episode_indices) == 96
    assert len(split.validation_episode_indices) == 24
    video_backend = os.environ.get("DREAMHANDOFF_VIDEO_BACKEND")
    result = train_world_model(
        R2DreamerTrainingConfig(
            dataset_root=dataset,
            split_path=split_path,
            output_dir=tmp_path / "run",
            device="cuda",
            replay_storage_device="cpu",
            steps=2,
            validation_interval=2,
            checkpoint_interval=2,
            validation_batches=1,
            cache_dir=tmp_path / "cache",
            video_backend=video_backend,
        )
    )
    runtime = R2DreamerRuntime.from_checkpoint(result.latest_checkpoint, "cuda")
    assert runtime.model.config.action_dim == 6
    assert result.final_metrics["step"] == 2
    assert "validation/total" in result.final_metrics

    resumed = train_world_model(
        R2DreamerTrainingConfig(
            dataset_root=dataset,
            split_path=split_path,
            output_dir=tmp_path / "resumed-run",
            device="cuda",
            replay_storage_device="cpu",
            steps=3,
            validation_interval=3,
            checkpoint_interval=3,
            validation_batches=1,
            cache_dir=tmp_path / "cache",
            resume_path=result.latest_checkpoint,
            video_backend=video_backend,
        )
    )
    resumed_runtime = R2DreamerRuntime.from_checkpoint(resumed.latest_checkpoint, "cuda")
    assert resumed_runtime.model.config.action_dim == 6
    assert resumed.final_metrics["step"] == 3
