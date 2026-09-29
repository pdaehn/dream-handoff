from __future__ import annotations

import os
from pathlib import Path

import pytest
from conftest import require_existing_directory

from dream_handoff.dataset import (
    FINAL_DATASET_HF_REVISION,
    FINAL_DATASET_REPO_ID,
    FINAL_EPISODE_METADATA_SHA256,
    FINAL_SPLIT_SCIENTIFIC_SHA256,
    FINAL_SPLIT_SHA256,
    FINAL_TRAIN_FRAMES,
    EpisodeSplit,
    create_episode_split,
    create_final_episode_split,
    episode_metadata_sha256,
    read_lerobot_episode_metadata,
    training_frame_count,
)

ROOT = Path(__file__).parents[1]
SPLIT_PATH = ROOT / "configs/data/episode_split.json"
EXPECTED_VALIDATION = (
    79,
    12,
    96,
    101,
    36,
    17,
    64,
    27,
    74,
    45,
    61,
    38,
    106,
    100,
    51,
    62,
    65,
    33,
    5,
    53,
    113,
    97,
    49,
    108,
)


def test_checked_in_final_split_has_exact_identity_and_partition() -> None:
    split = EpisodeSplit.read(SPLIT_PATH)

    assert len(split.train_episode_indices) == 96
    assert len(split.validation_episode_indices) == 24
    assert split.validation_episode_indices == EXPECTED_VALIDATION
    assert not set(split.train_episode_indices).intersection(split.validation_episode_indices)
    assert set((*split.train_episode_indices, *split.validation_episode_indices)) == set(range(120))
    assert split.episode_metadata_sha256 == FINAL_EPISODE_METADATA_SHA256
    assert split.canonical_sha256 == FINAL_SPLIT_SCIENTIFIC_SHA256
    assert split.dataset_repo_id == "pdaehn/dreamhandoff-so101-rectangle-on-peg"
    assert FINAL_SPLIT_SHA256 == (
        "cd0893cdbfd2d65198f8de8e55c8505b89bf40f998c3f846d50d4a42a03ffa21"
    )


def test_scientific_split_hash_excludes_host_locator() -> None:
    split = EpisodeSplit.read(SPLIT_PATH)
    alternative = EpisodeSplit.from_dict(
        {**split.to_dict(), "dataset_repo_id": "example/alternate"}
    )
    assert alternative.canonical_sha256 == split.canonical_sha256


def test_task_stratified_split_is_seed_reproducible_on_small_fixture() -> None:
    tasks = (("task-a",),) * 5 + (("task-b",),) * 2
    lengths = (10, 11, 12, 13, 14, 15, 16)
    arguments = {
        "dataset_id": "fixture",
        "dataset_repo_id": "example/fixture",
        "dataset_revision": None,
        "episode_tasks": tasks,
        "episode_lengths": lengths,
        "validation_fraction": 0.4,
        "seed": 0,
    }

    first = create_episode_split(**arguments)
    repeated = create_episode_split(**arguments)

    assert first == repeated
    assert first.train_episode_indices == (2, 1, 0, 5)
    assert first.validation_episode_indices == (4, 3, 6)
    assert not set(first.train_episode_indices).intersection(first.validation_episode_indices)
    assert set((*first.train_episode_indices, *first.validation_episode_indices)) == set(range(7))


def test_episode_metadata_hash_detects_task_or_length_changes() -> None:
    baseline = episode_metadata_sha256((("task",), ("task",)), (10, 11))

    assert episode_metadata_sha256((("task",), ("other",)), (10, 11)) != baseline
    assert episode_metadata_sha256((("task",), ("task",)), (10, 12)) != baseline


def test_training_frame_count_uses_only_explicit_episode_membership() -> None:
    lengths = (3, 5, 11, 17)

    assert training_frame_count(lengths, (2, 0)) == 14
    assert training_frame_count(lengths, (3, 1)) == 22


def test_real_dataset_gate_rejects_incomplete_local_artifact(
    tmp_path: Path, local_lerobot_dataset: None
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    with pytest.raises(pytest.fail.Exception, match="local dataset is incomplete"):
        LeRobotDataset(
            FINAL_DATASET_REPO_ID,
            root=tmp_path,
            revision=FINAL_DATASET_HF_REVISION,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.dataset_artifact
def test_recovered_dataset_regenerates_checked_in_manifest_exactly(
    local_lerobot_dataset: None,
) -> None:
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.sampler import EpisodeAwareSampler

    from dream_handoff.r2dreamer.data import verify_recovered_dataset_content

    configured = os.environ.get("DREAMHANDOFF_DATASET")
    if configured is None:
        pytest.skip("set DREAMHANDOFF_DATASET to run the real-dataset split gate")
    dataset_root = require_existing_directory(configured, "DREAMHANDOFF_DATASET")

    regenerated = create_final_episode_split(dataset_root)
    checked_in = EpisodeSplit.read(SPLIT_PATH)

    assert regenerated.to_dict() == checked_in.to_dict()
    assert regenerated.validation_episode_indices == EXPECTED_VALIDATION
    assert regenerated.episode_metadata_sha256 == FINAL_EPISODE_METADATA_SHA256
    assert regenerated.canonical_sha256 == FINAL_SPLIT_SCIENTIFIC_SHA256

    verify_recovered_dataset_content(dataset_root)
    _, episode_lengths = read_lerobot_episode_metadata(dataset_root)
    assert training_frame_count(episode_lengths, checked_in.train_episode_indices) == (
        FINAL_TRAIN_FRAMES
    )

    selected = set(checked_in.train_episode_indices)
    parquet_rows = sum(
        sum(int(index) in selected for index in pq.read_table(path, columns=["episode_index"])[0])
        for path in dataset_root.glob("data/*/*.parquet")
    )
    assert parquet_rows == FINAL_TRAIN_FRAMES

    dataset = LeRobotDataset(
        checked_in.dataset_repo_id,
        root=dataset_root,
        episodes=list(checked_in.train_episode_indices),
        revision=checked_in.dataset_revision,
    )
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        shuffle=True,
        seed=0,
        absolute_to_relative_idx=dataset.absolute_to_relative_idx,
    )
    assert len(dataset) == len(sampler) == FINAL_TRAIN_FRAMES
