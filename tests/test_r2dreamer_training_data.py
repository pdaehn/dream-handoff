from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import dream_handoff.dataset as dataset_module
import dream_handoff.r2dreamer.data as data_module
from dream_handoff.r2dreamer.action_normalization import R2DreamerActionNormalizer
from dream_handoff.r2dreamer.config import R2DreamerModelConfig, R2DreamerPreprocessingConfig
from dream_handoff.r2dreamer.data import (
    FINAL_EPISODE_METADATA_SHA256,
    FINAL_SPLIT_SCIENTIFIC_SHA256,
    EpisodeSequenceDataset,
    episode_metadata_sha256,
    read_final_episode_split,
    verify_recovered_dataset_content,
)
from dream_handoff.r2dreamer.preprocessing import R2DreamerObservationPreprocessor
from dream_handoff.r2dreamer.replay import EpisodeTensorStore, materialize_episode_dataset

TRAIN = (
    57,
    103,
    99,
    3,
    111,
    59,
    73,
    22,
    30,
    25,
    47,
    69,
    23,
    67,
    75,
    16,
    85,
    29,
    2,
    76,
    8,
    107,
    43,
    84,
    98,
    44,
    46,
    115,
    80,
    37,
    10,
    24,
    48,
    50,
    104,
    93,
    13,
    52,
    21,
    112,
    72,
    91,
    35,
    19,
    6,
    102,
    95,
    20,
    118,
    114,
    28,
    34,
    54,
    88,
    94,
    15,
    14,
    109,
    58,
    83,
    4,
    81,
    82,
    41,
    31,
    86,
    63,
    0,
    110,
    11,
    1,
    92,
    7,
    116,
    66,
    56,
    119,
    70,
    26,
    78,
    40,
    55,
    105,
    89,
    71,
    60,
    42,
    87,
    9,
    117,
    39,
    18,
    77,
    90,
    68,
    32,
)
VALIDATION = (
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


def split_payload() -> dict:
    return {
        "dataset_id": "dreamhandoff-so101-rectangle-on-peg",
        "dataset_repo_id": "pdaehn/dreamhandoff-so101-rectangle-on-peg",
        "dataset_revision": data_module.FINAL_DATASET_HF_REVISION,
        "episode_metadata_sha256": FINAL_EPISODE_METADATA_SHA256,
        "schema_version": 1,
        "seed": 0,
        "strategy": "task-stratified",
        "total_episodes": 120,
        "train_episode_indices": list(TRAIN),
        "validation_episode_indices": list(VALIDATION),
        "validation_fraction": 0.2,
    }


def test_immutable_final_split_membership_and_canonical_hash(tmp_path: Path) -> None:
    path = tmp_path / "episode_split.json"
    path.write_text(json.dumps(split_payload(), indent=2))

    split = read_final_episode_split(path)

    assert split.train_episode_indices == TRAIN
    assert split.validation_episode_indices == VALIDATION
    assert len(split.train_episode_indices) == 96
    assert len(split.validation_episode_indices) == 24
    assert set(split.train_episode_indices).isdisjoint(split.validation_episode_indices)
    assert split.split_sha256 == FINAL_SPLIT_SCIENTIFIC_SHA256


def test_r2_uses_canonical_dataset_provenance_definitions() -> None:
    assert data_module.FINAL_DATASET_PAYLOAD_FILES is dataset_module.FINAL_DATASET_PAYLOAD_FILES
    assert data_module.episode_metadata_sha256 is dataset_module.episode_metadata_sha256
    assert data_module.read_final_episode_split is dataset_module.read_final_episode_split
    assert (
        data_module.verify_recovered_dataset_content
        is dataset_module.verify_recovered_dataset_content
    )


def test_split_provenance_mismatch_fails_clearly(tmp_path: Path) -> None:
    payload = split_payload()
    payload["dataset_repo_id"] = "someone/another-dataset"
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="dataset_repo_id does not match"):
        read_final_episode_split(path)


def test_split_rejects_different_revision(tmp_path: Path) -> None:
    payload = split_payload()
    payload["dataset_revision"] = "0" * 40
    path = tmp_path / "rewritten-revision.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="dataset_revision does not match"):
        read_final_episode_split(path)


def test_split_episode_order_mismatch_fails_canonical_identity(tmp_path: Path) -> None:
    payload = split_payload()
    payload["train_episode_indices"][:2] = reversed(payload["train_episode_indices"][:2])
    path = tmp_path / "reordered.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="scientific SHA-256"):
        read_final_episode_split(path)


def test_split_episode_metadata_mismatch_fails_clearly(tmp_path: Path) -> None:
    payload = split_payload()
    payload["episode_metadata_sha256"] = "0" * 64
    path = tmp_path / "wrong-metadata.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="episode_metadata_sha256 does not match"):
        read_final_episode_split(path)


def test_recovered_revision_content_verification_hashes_local_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = "data/chunk-000/file-000.parquet"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"recovered bytes")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(dataset_module, "FINAL_DATASET_PAYLOAD_FILES", ((relative, expected),))

    (tmp_path / "README.md").write_text("First card\n")
    (tmp_path / ".gitattributes").write_text("*.parquet filter=lfs\n")
    verify_recovered_dataset_content(tmp_path)
    (tmp_path / "README.md").write_text("Updated card\n")
    verify_recovered_dataset_content(tmp_path)
    extra = tmp_path / "meta/extra.json"
    extra.parent.mkdir(parents=True)
    extra.write_text("{}")
    with pytest.raises(ValueError, match="unclassified files"):
        verify_recovered_dataset_content(tmp_path)
    extra.unlink()
    path.write_bytes(b"different local bytes")

    with pytest.raises(ValueError, match="does not match release revision"):
        verify_recovered_dataset_content(tmp_path)


class Frames:
    def __init__(self) -> None:
        self.rows = []
        for episode, length in ((0, 5), (1, 4)):
            for frame in range(length):
                value = episode * 100 + frame
                self.rows.append(
                    {
                        "episode_index": episode,
                        "frame_index": frame,
                        "observation.images.context": torch.full(
                            (3, 8, 8), value, dtype=torch.uint8
                        ),
                        "observation.images.wrist": torch.full(
                            (3, 8, 8), value + 1, dtype=torch.uint8
                        ),
                        "observation.state": torch.full((6,), float(value)),
                        "action": torch.full((6,), float(value + 10)),
                    }
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def sequence_dataset() -> EpisodeSequenceDataset:
    return EpisodeSequenceDataset(
        Frames(),
        sequence_length=3,
        image_keys=("observation.images.context", "observation.images.wrist"),
        state_key="observation.state",
        episode_indices=(0, 1),
    )


def test_windows_keep_frozen_previous_action_alignment_and_boundaries() -> None:
    dataset = sequence_dataset()
    assert len(dataset) == 5
    first, middle, last = dataset[0], dataset[1], dataset[-1]
    assert first["frame_index"].tolist() == [0, 1, 2]
    assert middle["frame_index"].tolist() == [1, 2, 3]
    assert last["frame_index"].tolist() == [1, 2, 3]
    assert int(last["episode_index"]) == 1
    assert first["previous_action"][:, 0].tolist() == [0.0, 10.0, 11.0]
    assert middle["previous_action"][:, 0].tolist() == [10.0, 11.0, 12.0]
    assert last["previous_action"][:, 0].tolist() == [110.0, 111.0, 112.0]
    assert first["is_first"].tolist() == [True, False, False]
    assert middle["is_first"].tolist() == [False, False, False]


def test_batch_materialization_reuses_wp2a_preprocessing_and_normalization() -> None:
    dataset = sequence_dataset()
    preprocessing = R2DreamerPreprocessingConfig(image_size=(4, 4))
    normalizer = R2DreamerActionNormalizer(
        action_names=tuple(f"joint_{index}" for index in range(6)),
        q01=(0.0,) * 6,
        q99=(120.0,) * 6,
    )
    payload = materialize_episode_dataset(dataset, preprocessing, normalizer)
    prepared = R2DreamerObservationPreprocessor(preprocessing)(dataset[0])

    assert prepared["images"].shape == (3, 4, 4, 6)
    assert prepared["state"].shape == (3, 6)
    assert payload["observations"]["images"].shape == (9, 4, 4, 6)
    assert payload["previous_actions"].shape == (9, 6)
    assert payload["previous_actions"][0].eq(0).all()
    torch.testing.assert_close(payload["previous_actions"][1], torch.full((6,), -5 / 6))


def test_replay_samples_valid_starts_seededly_and_uses_preceding_latent() -> None:
    payload = materialize_episode_dataset(
        sequence_dataset(),
        R2DreamerPreprocessingConfig(image_size=(4, 4)),
        R2DreamerActionNormalizer(
            action_names=tuple(f"joint_{index}" for index in range(6)),
            q01=(0.0,) * 6,
            q99=(120.0,) * 6,
        ),
    )
    config = R2DreamerModelConfig(
        action_dim=6,
        image_channels=6,
        state_dim=6,
        image_size=(4, 4),
        stoch=2,
        deter=8,
        hidden=4,
        discrete=2,
        blocks=2,
        encoder_depth=2,
        encoder_mults=(1, 1),
        vector_layers=1,
        vector_units=4,
    )
    first = EpisodeTensorStore(
        payload, sequence_length=3, model_config=config, device="cpu", storage_device="cpu", seed=7
    )
    second = EpisodeTensorStore(
        payload, sequence_length=3, model_config=config, device="cpu", storage_device="cpu", seed=7
    )
    first.latent_valid.fill_(True)
    second.latent_valid.fill_(True)
    first.cached_deter[0].fill_(3)
    assert first.next_batch(5).row_indices.tolist() == second.next_batch(5).row_indices.tolist()
    mid = first.batch_from_window_indices(torch.tensor([1]))
    torch.testing.assert_close(mid.initial_state.deter, torch.full((1, 8), 3.0))


def test_episode_metadata_hash_is_sensitive_to_dataset_provenance() -> None:
    original = episode_metadata_sha256([("task",), ("task",)], [5, 4])
    changed = episode_metadata_sha256([("task",), ("task",)], [5, 5])
    assert original != changed
