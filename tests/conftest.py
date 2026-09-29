from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def local_lerobot_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real-data gates read-only when a local dataset is incomplete."""
    from lerobot.datasets import dataset_metadata, lerobot_dataset

    def reject_download(*_args: object, **_kwargs: object) -> None:
        pytest.fail(
            "local dataset is incomplete; download the pinned artifact before running this gate"
        )

    monkeypatch.setattr(dataset_metadata, "get_safe_version", reject_download)
    monkeypatch.setattr(dataset_metadata.LeRobotDatasetMetadata, "_pull_from_repo", reject_download)
    monkeypatch.setattr(lerobot_dataset, "get_safe_version", reject_download)
    monkeypatch.setattr(lerobot_dataset.LeRobotDataset, "_download", reject_download)


def require_existing_directory(value: str, name: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        pytest.fail(f"{name} does not exist: {path}")
    return path
