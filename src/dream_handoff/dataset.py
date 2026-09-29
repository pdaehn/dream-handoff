"""Deterministic episode splitting for the final DreamHandoff dataset."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FINAL_DATASET_ID = "dreamhandoff-so101-rectangle-on-peg"
FINAL_DATASET_REPO_ID = "pdaehn/dreamhandoff-so101-rectangle-on-peg"
FINAL_DATASET_HF_REVISION = "70d08de50664a8c973375484c56ca553815719e5"
FINAL_EPISODE_METADATA_SHA256 = "7ef860f074a7687405a985a1d1cd7ccc04df01c1e6f50dcbcaaab4309a117555"
FINAL_SPLIT_SHA256 = "cd0893cdbfd2d65198f8de8e55c8505b89bf40f998c3f846d50d4a42a03ffa21"
FINAL_SPLIT_SCIENTIFIC_SHA256 = "e7d93ce2f0c5f7ae60f1dfcdfb2cbce68ff194e1d559e90145f61b97795f6f5a"
FINAL_TRAIN_FRAMES = 46_690
FINAL_DATASET_PAYLOAD_FILES = (
    (
        "data/chunk-000/file-000.parquet",
        "7014059440c5c273eb0ce692372713e000e09e285f015fa4d8a9b8ca052eaae6",
    ),
    (
        "meta/episodes/chunk-000/file-000.parquet",
        "7a3a6baabf5605624f94513a65a2b5858e3bf25f57157f6dd5bd31310defcbbc",
    ),
    ("meta/info.json", "7931d579e6c75a1707794f6e2e85c6c6af9b0a716c8d3f2000cd4a91c1a863d9"),
    ("meta/stats.json", "6ff351c83f8c30498103d0953e01e659cb28555101ac7c5b9d4259dc9cdd2ecd"),
    (
        "meta/tasks.parquet",
        "e30a96a1d7183de8b0720d3ee4d9ce2c80835b12aacbd993f7ba34bd7cf73da7",
    ),
    (
        "videos/observation.images.context/chunk-000/file-000.mp4",
        "7f23b02093afef4897ee1a2a0a047b8c636b26da45e5c3c0a3221499e20dfc7f",
    ),
    (
        "videos/observation.images.wrist/chunk-000/file-000.mp4",
        "d6bb7035674ac43091f0e803a820bdfcaa0d3c0aff762fab987c7ea592485ca9",
    ),
    (
        "videos/observation.images.wrist/chunk-000/file-001.mp4",
        "9d8b1d94ba414b0b3cda28fefc483ccd4a5fa939900b698300450731aca4297c",
    ),
)
FINAL_DATASET_REPOSITORY_METADATA_FILES = frozenset({"README.md", ".gitattributes"})


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for scientific identity checks."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON content independently of file whitespace and key ordering."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def raw_file_sha256(path: Path | str) -> str:
    """Hash the bytes of a file; this is distinct from the canonical JSON digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_recovered_dataset_content(dataset_root: Path | str) -> None:
    """Verify all dataset data and loading metadata, excluding presentation files."""

    root = Path(dataset_root)
    expected_files = {relative_path for relative_path, _ in FINAL_DATASET_PAYLOAD_FILES}
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).parts[0] != ".cache"
    }
    unexpected = actual_files - expected_files - FINAL_DATASET_REPOSITORY_METADATA_FILES
    if unexpected:
        raise ValueError(f"dataset release contains unclassified files: {sorted(unexpected)}")
    for relative_path, expected in FINAL_DATASET_PAYLOAD_FILES:
        path = root / relative_path
        try:
            actual = raw_file_sha256(path)
        except OSError as exc:
            raise ValueError(
                f"dataset release revision {FINAL_DATASET_HF_REVISION} is missing "
                f"{relative_path}: {exc}"
            ) from exc
        if actual != expected:
            raise ValueError(
                f"dataset payload {relative_path} does not match release revision "
                f"{FINAL_DATASET_HF_REVISION} ({actual} != {expected})"
            )


def episode_metadata_sha256(
    episode_tasks: Sequence[Sequence[str]], episode_lengths: Sequence[int]
) -> str:
    """Fingerprint the ordered episode namespace, primary data, and lengths."""
    if len(episode_tasks) != len(episode_lengths):
        raise ValueError("episode task and length metadata must have the same size")
    rows = [
        {
            "episode_index": episode_index,
            "tasks": [str(task) for task in tasks],
            "length": int(episode_lengths[episode_index]),
        }
        for episode_index, tasks in enumerate(episode_tasks)
    ]
    return canonical_json_sha256(rows)


def training_frame_count(
    episode_lengths: Sequence[int], train_episode_indices: Sequence[int]
) -> int:
    """Count frames in an explicit episode-level training partition."""
    return sum(int(episode_lengths[int(index)]) for index in train_episode_indices)


def validate_episode_partition(
    train_episode_indices: Sequence[int],
    validation_episode_indices: Sequence[int],
    *,
    total_episodes: int,
) -> None:
    """Require a complete, disjoint, duplicate-free episode partition."""
    train = tuple(int(index) for index in train_episode_indices)
    validation = tuple(int(index) for index in validation_episode_indices)
    if total_episodes < 2 or not train or not validation:
        raise ValueError("an episode split requires non-empty train and validation partitions")
    if len(set(train)) != len(train):
        raise ValueError("training episode indices contain duplicates")
    if len(set(validation)) != len(validation):
        raise ValueError("validation episode indices contain duplicates")
    if overlap := sorted(set(train).intersection(validation)):
        raise ValueError(f"training and validation episodes overlap: {overlap}")
    expected = set(range(total_episodes))
    actual = set(train).union(validation)
    if actual != expected:
        raise ValueError(
            "episode split does not exactly partition the dataset "
            f"(missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)})"
        )


def generate_episode_indices(
    episode_tasks: Sequence[Sequence[str]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Reproduce the final task-stratified split algorithm and ordering."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if len(episode_tasks) < 2:
        raise ValueError("an episode split requires at least two episodes")

    task_to_episodes: dict[str, list[int]] = {}
    for episode_index, tasks in enumerate(episode_tasks):
        primary_task = str(tasks[0]) if tasks else ""
        task_to_episodes.setdefault(primary_task, []).append(episode_index)

    train: list[int] = []
    validation: list[int] = []
    for task_episodes in task_to_episodes.values():
        # The frozen producer deliberately restarted the same seeded stream for
        # each task. Preserve this detail because it determines manifest order.
        random.Random(seed).shuffle(task_episodes)
        validation_count = math.ceil(len(task_episodes) * validation_fraction)
        train.extend(task_episodes[:-validation_count])
        validation.extend(task_episodes[-validation_count:])

    validate_episode_partition(train, validation, total_episodes=len(episode_tasks))
    return tuple(train), tuple(validation)


@dataclass(frozen=True)
class EpisodeSplit:
    """The schema-1 split document consumed by the final training runs."""

    dataset_id: str
    dataset_repo_id: str
    dataset_revision: str | None
    total_episodes: int
    episode_metadata_sha256: str
    strategy: str
    validation_fraction: float
    seed: int
    train_episode_indices: tuple[int, ...]
    validation_episode_indices: tuple[int, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "train_episode_indices", tuple(int(i) for i in self.train_episode_indices)
        )
        object.__setattr__(
            self,
            "validation_episode_indices",
            tuple(int(i) for i in self.validation_episode_indices),
        )
        if self.schema_version != 1:
            raise ValueError("only the final schema-1 split is supported")
        if self.strategy != "task-stratified":
            raise ValueError("only the final task-stratified strategy is supported")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if not self.dataset_id.strip() or not self.dataset_repo_id.strip():
            raise ValueError("dataset_id and dataset_repo_id must not be empty")
        if len(self.episode_metadata_sha256) != 64:
            raise ValueError("episode_metadata_sha256 must be a SHA-256 hex digest")
        try:
            int(self.episode_metadata_sha256, 16)
        except ValueError as exc:
            raise ValueError("episode_metadata_sha256 must be a SHA-256 hex digest") from exc
        validate_episode_partition(
            self.train_episode_indices,
            self.validation_episode_indices,
            total_episodes=self.total_episodes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_repo_id": self.dataset_repo_id,
            "dataset_revision": self.dataset_revision,
            "total_episodes": self.total_episodes,
            "episode_metadata_sha256": self.episode_metadata_sha256,
            "strategy": self.strategy,
            "validation_fraction": self.validation_fraction,
            "seed": self.seed,
            "train_episode_indices": list(self.train_episode_indices),
            "validation_episode_indices": list(self.validation_episode_indices),
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_json_sha256(
            {
                "episode_metadata_sha256": self.episode_metadata_sha256,
                "strategy": self.strategy,
                "validation_fraction": self.validation_fraction,
                "seed": self.seed,
                "total_episodes": self.total_episodes,
                "train_episode_indices": self.train_episode_indices,
                "validation_episode_indices": self.validation_episode_indices,
            }
        )

    @property
    def split_sha256(self) -> str:
        """Compatibility name used by the R2 training adapter."""

        return self.canonical_sha256

    def write(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EpisodeSplit:
        return cls(
            schema_version=int(value["schema_version"]),
            dataset_id=str(value["dataset_id"]),
            dataset_repo_id=str(value["dataset_repo_id"]),
            dataset_revision=(
                None if value.get("dataset_revision") is None else str(value["dataset_revision"])
            ),
            total_episodes=int(value["total_episodes"]),
            episode_metadata_sha256=str(value["episode_metadata_sha256"]),
            strategy=str(value["strategy"]),
            validation_fraction=float(value["validation_fraction"]),
            seed=int(value["seed"]),
            train_episode_indices=tuple(value["train_episode_indices"]),
            validation_episode_indices=tuple(value["validation_episode_indices"]),
        )

    @classmethod
    def read(cls, path: Path | str) -> EpisodeSplit:
        source = Path(path)
        try:
            value = json.loads(source.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read episode split {source}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"episode split {source} must contain a JSON object")
        try:
            return cls.from_dict(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid episode split {source}: {exc}") from exc


def read_final_episode_split(path: Path | str) -> EpisodeSplit:
    """Read and strictly validate the final 96/24 scientific partition."""

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read R2 episode split {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("R2 episode split must contain a JSON object")

    expected_fields = {
        "schema_version": 1,
        "dataset_id": FINAL_DATASET_ID,
        "dataset_repo_id": FINAL_DATASET_REPO_ID,
        "dataset_revision": FINAL_DATASET_HF_REVISION,
        "total_episodes": 120,
        "episode_metadata_sha256": FINAL_EPISODE_METADATA_SHA256,
        "strategy": "task-stratified",
        "validation_fraction": 0.2,
        "seed": 0,
    }
    for name, expected in expected_fields.items():
        if value.get(name) != expected:
            raise ValueError(
                f"R2 split {name} does not match the final dataset "
                f"({value.get(name)!r} != {expected!r})"
            )

    try:
        split = EpisodeSplit.from_dict(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid R2 episode split {source}: {exc}") from exc
    if len(split.train_episode_indices) != 96 or len(split.validation_episode_indices) != 24:
        raise ValueError("final R2 split must contain 96 train and 24 validation episodes")
    if split.canonical_sha256 != FINAL_SPLIT_SCIENTIFIC_SHA256:
        raise ValueError(
            "R2 split scientific SHA-256 mismatch "
            f"({split.canonical_sha256} != {FINAL_SPLIT_SCIENTIFIC_SHA256})"
        )
    return split


def create_episode_split(
    *,
    dataset_id: str,
    dataset_repo_id: str,
    dataset_revision: str | None,
    episode_tasks: Sequence[Sequence[str]],
    episode_lengths: Sequence[int],
    validation_fraction: float,
    seed: int,
) -> EpisodeSplit:
    """Build a manifest from ordered LeRobot episode metadata."""
    train, validation = generate_episode_indices(
        episode_tasks, validation_fraction=validation_fraction, seed=seed
    )
    return EpisodeSplit(
        dataset_id=dataset_id,
        dataset_repo_id=dataset_repo_id,
        dataset_revision=dataset_revision,
        total_episodes=len(episode_tasks),
        episode_metadata_sha256=episode_metadata_sha256(episode_tasks, episode_lengths),
        strategy="task-stratified",
        validation_fraction=validation_fraction,
        seed=seed,
        train_episode_indices=train,
        validation_episode_indices=validation,
    )


def read_lerobot_episode_metadata(
    dataset_root: Path | str,
) -> tuple[tuple[tuple[str, ...], ...], tuple[int, ...]]:
    """Read ordered episode parquet metadata from a local LeRobot dataset."""
    from lerobot.datasets.io_utils import load_episodes

    episodes = load_episodes(Path(dataset_root))
    indices = tuple(int(index) for index in episodes["episode_index"])
    if indices != tuple(range(len(indices))):
        raise ValueError(
            "dataset episode metadata is not ordered in a contiguous zero-based namespace"
        )
    tasks = tuple(tuple(str(task) for task in row) for row in episodes["tasks"])
    lengths = tuple(int(length) for length in episodes["length"])
    return tasks, lengths


def create_final_episode_split(dataset_root: Path | str) -> EpisodeSplit:
    """Regenerate the final manifest from a local copy of the dataset."""
    tasks, lengths = read_lerobot_episode_metadata(dataset_root)
    split = create_episode_split(
        dataset_id=FINAL_DATASET_ID,
        dataset_repo_id=FINAL_DATASET_REPO_ID,
        dataset_revision=FINAL_DATASET_HF_REVISION,
        episode_tasks=tasks,
        episode_lengths=lengths,
        validation_fraction=0.2,
        seed=0,
    )
    if split.episode_metadata_sha256 != FINAL_EPISODE_METADATA_SHA256:
        raise ValueError(
            "dataset episode metadata SHA-256 does not match the final dataset "
            f"({split.episode_metadata_sha256} != {FINAL_EPISODE_METADATA_SHA256})"
        )
    if split.canonical_sha256 != FINAL_SPLIT_SCIENTIFIC_SHA256:
        raise ValueError(
            "regenerated split SHA-256 does not match the final manifest "
            f"({split.canonical_sha256} != {FINAL_SPLIT_SCIENTIFIC_SHA256})"
        )
    return split
