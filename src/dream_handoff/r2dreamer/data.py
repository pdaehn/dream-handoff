"""Final-split LeRobot sequence adaptation for offline R2 training."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch.utils.data import Dataset

from dream_handoff.dataset import (
    FINAL_DATASET_HF_REVISION,
    EpisodeSplit,
    episode_metadata_sha256,
    verify_recovered_dataset_content,
)
from dream_handoff.dataset import FINAL_DATASET_ID as FINAL_DATASET_ID
from dream_handoff.dataset import FINAL_DATASET_PAYLOAD_FILES as FINAL_DATASET_PAYLOAD_FILES
from dream_handoff.dataset import FINAL_DATASET_REPO_ID as FINAL_DATASET_REPO_ID
from dream_handoff.dataset import (
    FINAL_EPISODE_METADATA_SHA256 as FINAL_EPISODE_METADATA_SHA256,
)
from dream_handoff.dataset import FINAL_SPLIT_SCIENTIFIC_SHA256 as FINAL_SPLIT_SCIENTIFIC_SHA256
from dream_handoff.dataset import read_final_episode_split as read_final_episode_split

from .action_normalization import R2DreamerActionNormalizer
from .preprocessing import prepare_image_tensor


class FrameDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> dict[str, Any]: ...


FinalEpisodeSplit = EpisodeSplit


def _integer(value: Any) -> int:
    return int(value.item()) if torch.is_tensor(value) else int(value)


def _column_source(frames: FrameDataset) -> Any:
    return getattr(frames, "hf_dataset", frames)


@dataclass(frozen=True)
class EpisodeWindow:
    episode_index: int
    start_offset: int
    row_indices: tuple[int, ...]


class EpisodeSequenceDataset(Dataset[dict[str, Any]]):
    """Return episode-local ``o_t`` windows paired with canonical ``a_(t-1)``."""

    def __init__(
        self,
        frames: FrameDataset,
        *,
        sequence_length: int,
        image_keys: Sequence[str],
        state_key: str | None,
        action_key: str = "action",
        episode_indices: Sequence[int],
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.frames = frames
        self.sequence_length = sequence_length
        self.image_keys = tuple(image_keys)
        self.state_key = state_key
        self.action_key = action_key
        selected = set(int(index) for index in episode_indices)
        episode_rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for row_index, (episode_index, frame_index) in enumerate(self._frame_metadata(frames)):
            if episode_index in selected:
                episode_rows[episode_index].append((frame_index, row_index))
        if set(episode_rows) != selected:
            missing = sorted(selected.difference(episode_rows))
            raise ValueError(f"dataset has no frames for split episodes: {missing}")

        self.episode_rows: dict[int, tuple[int, ...]] = {}
        windows: list[EpisodeWindow] = []
        for episode_index in sorted(episode_rows):
            indexed_rows = sorted(episode_rows[episode_index])
            frame_indices = [frame for frame, _ in indexed_rows]
            if frame_indices != list(range(len(frame_indices))):
                raise ValueError(f"episode {episode_index} has non-contiguous zero-based frames")
            rows = tuple(row for _, row in indexed_rows)
            self.episode_rows[episode_index] = rows
            windows.extend(
                EpisodeWindow(episode_index, start, rows[start : start + sequence_length])
                for start in range(len(rows) - sequence_length + 1)
            )
        self.windows = tuple(windows)

    @staticmethod
    def _frame_metadata(frames: FrameDataset) -> list[tuple[int, int]]:
        source = _column_source(frames)
        select_columns = getattr(source, "select_columns", None)
        if callable(select_columns):
            columns = select_columns(["episode_index", "frame_index"])
            return [
                (_integer(episode), _integer(frame))
                for episode, frame in zip(
                    columns["episode_index"], columns["frame_index"], strict=True
                )
            ]
        return [
            (_integer(frames[index]["episode_index"]), _integer(frames[index]["frame_index"]))
            for index in range(len(frames))
        ]

    @property
    def episode_indices(self) -> tuple[int, ...]:
        return tuple(self.episode_rows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        observations = [self.frames[row] for row in window.row_indices]
        current_actions = [torch.as_tensor(frame[self.action_key]) for frame in observations]
        if window.start_offset == 0:
            first_previous_action = torch.zeros_like(current_actions[0])
        else:
            preceding_row = self.episode_rows[window.episode_index][window.start_offset - 1]
            first_previous_action = torch.as_tensor(self.frames[preceding_row][self.action_key])
        result = {
            key: torch.stack([torch.as_tensor(frame[key]) for frame in observations])
            for key in self.image_keys
        }
        if self.state_key is not None:
            result[self.state_key] = torch.stack(
                [torch.as_tensor(frame[self.state_key]) for frame in observations]
            )
        is_first = torch.zeros(self.sequence_length, dtype=torch.bool)
        if window.start_offset == 0:
            is_first[0] = True
        result.update(
            previous_action=torch.stack([first_previous_action, *current_actions[:-1]]).float(),
            is_first=is_first,
            episode_index=torch.tensor(window.episode_index),
            frame_index=torch.tensor([_integer(frame["frame_index"]) for frame in observations]),
        )
        return result


class LeRobotEpisodeSequenceDataset(EpisodeSequenceDataset):
    """Use one delta-timestamp LeRobot lookup per window and resize before collation."""

    def __init__(self, *args: Any, image_size: tuple[int, int], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.image_size = image_size

    @staticmethod
    def _padding(item: dict[str, Any], key: str, *, allow_first: bool = False) -> torch.Tensor:
        padding_key = f"{key}_is_pad"
        if padding_key not in item:
            raise RuntimeError(f"LeRobot sequence item is missing {padding_key!r}")
        padding = torch.as_tensor(item[padding_key], dtype=torch.bool)
        allowed = torch.zeros_like(padding)
        if allow_first:
            allowed[0] = True
        if bool((padding & ~allowed).any()):
            raise RuntimeError(f"LeRobot padded a valid episode window for {key!r}")
        return padding

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        item = self.frames[window.row_indices[0]]
        if _integer(item["episode_index"]) != window.episode_index:
            raise RuntimeError("sequence unexpectedly crossed an episode boundary")
        result: dict[str, Any] = {}
        for key in self.image_keys:
            self._padding(item, key)
            image = prepare_image_tensor(item[key], key=key, image_size=self.image_size)
            if image.shape[0] != self.sequence_length:
                raise RuntimeError(f"LeRobot returned the wrong sequence length for {key!r}")
            result[key] = image.movedim(-1, -3)
        if self.state_key is not None:
            self._padding(item, self.state_key)
            state = torch.as_tensor(item[self.state_key])
            if state.shape[0] != self.sequence_length:
                raise RuntimeError("LeRobot returned the wrong vector-state sequence length")
            result[self.state_key] = state
        action_padding = self._padding(item, self.action_key, allow_first=window.start_offset == 0)
        previous_actions = torch.as_tensor(item[self.action_key], dtype=torch.float32).clone()
        if previous_actions.shape[0] != self.sequence_length:
            raise RuntimeError("LeRobot returned the wrong action sequence length")
        is_first = torch.zeros(self.sequence_length, dtype=torch.bool)
        if window.start_offset == 0:
            if not bool(action_padding[0]):
                raise RuntimeError("episode-first previous action was not marked as padded")
            previous_actions[0].zero_()
            is_first[0] = True
        elif bool(action_padding.any()):
            raise RuntimeError("mid-episode previous action was unexpectedly padded")
        first_frame = _integer(item["frame_index"])
        result.update(
            previous_action=previous_actions,
            is_first=is_first,
            episode_index=torch.tensor(window.episode_index),
            frame_index=torch.arange(first_frame, first_frame + self.sequence_length),
        )
        return result


def _raw_action_rows(dataset: EpisodeSequenceDataset) -> torch.Tensor:
    rows = [row for episode in dataset.episode_indices for row in dataset.episode_rows[episode]]
    source = _column_source(dataset.frames)
    select_columns = getattr(source, "select_columns", None)
    if callable(select_columns):
        column = select_columns([dataset.action_key])[dataset.action_key]
        try:
            actions = torch.as_tensor(column, dtype=torch.float32)
        except (TypeError, ValueError):
            actions = torch.stack([torch.as_tensor(value, dtype=torch.float32) for value in column])
        return actions[torch.tensor(rows)]
    get_raw_item = getattr(dataset.frames, "get_raw_item", None)
    item = get_raw_item if callable(get_raw_item) else dataset.frames.__getitem__
    return torch.stack(
        [torch.as_tensor(item(row)[dataset.action_key], dtype=torch.float32) for row in rows]
    )


def fit_action_normalizer(
    dataset: EpisodeSequenceDataset, *, action_names: tuple[str, ...]
) -> R2DreamerActionNormalizer:
    """Fit float64 q01/q99 over raw actions from the 96 training episodes."""
    actions = _raw_action_rows(dataset)
    if actions.ndim != 2 or actions.shape[-1] != len(action_names):
        raise ValueError("training actions do not match action metadata")
    if not bool(torch.isfinite(actions).all()):
        raise ValueError("training actions contain NaN or Inf")
    quantiles = torch.quantile(
        actions.double(), torch.tensor([0.01, 0.99], dtype=torch.float64), dim=0
    )
    return R2DreamerActionNormalizer(
        action_names=action_names,
        q01=tuple(float(value) for value in quantiles[0]),
        q99=tuple(float(value) for value in quantiles[1]),
    )


def load_final_lerobot_datasets(
    *,
    dataset_root: Path,
    split: EpisodeSplit,
    sequence_length: int,
    image_keys: Sequence[str],
    image_size: tuple[int, int],
    state_key: str | None,
    action_key: str = "action",
    video_backend: str | None = None,
) -> tuple[EpisodeSequenceDataset, EpisodeSequenceDataset, Any]:
    """Open the pinned LeRobot revision and verify content and split provenance."""
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    verify_recovered_dataset_content(dataset_root)
    metadata = LeRobotDatasetMetadata(
        repo_id=FINAL_DATASET_REPO_ID, root=dataset_root, revision=FINAL_DATASET_HF_REVISION
    )
    offsets = [offset / metadata.fps for offset in range(sequence_length)]
    delta_timestamps = {
        **{key: offsets for key in image_keys},
        **({state_key: offsets} if state_key is not None else {}),
        action_key: [(offset - 1) / metadata.fps for offset in range(sequence_length)],
    }
    dataset = LeRobotDataset(
        repo_id=FINAL_DATASET_REPO_ID,
        root=dataset_root,
        revision=FINAL_DATASET_HF_REVISION,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
        return_uint8=True,
    )
    total_episodes = len(dataset.meta.episodes["length"])
    if total_episodes != split.total_episodes:
        raise ValueError(
            f"dataset has {total_episodes} episodes but the immutable split requires 120"
        )
    actual_metadata = episode_metadata_sha256(
        dataset.meta.episodes["tasks"], dataset.meta.episodes["length"]
    )
    if actual_metadata != split.episode_metadata_sha256:
        raise ValueError(
            "dataset episode metadata does not match the immutable split "
            f"({actual_metadata} != {split.episode_metadata_sha256})"
        )
    available = sorted({episode for episode, _ in EpisodeSequenceDataset._frame_metadata(dataset)})
    if available != list(range(split.total_episodes)):
        raise ValueError("dataset frames do not contain the complete split episode namespace")
    common = {
        "frames": dataset,
        "sequence_length": sequence_length,
        "image_keys": image_keys,
        "image_size": image_size,
        "state_key": state_key,
        "action_key": action_key,
    }
    train = LeRobotEpisodeSequenceDataset(**common, episode_indices=split.train_episode_indices)
    validation = LeRobotEpisodeSequenceDataset(
        **common, episode_indices=split.validation_episode_indices
    )
    if not train or not validation:
        raise ValueError("an immutable split partition has no complete sequence windows")
    return train, validation, dataset
