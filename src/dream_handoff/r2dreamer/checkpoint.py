"""Strict loading and identity checks for R2-Dreamer checkpoints."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .action_normalization import R2DreamerActionNormalizer
from .config import (
    R2DreamerModelConfig,
    R2DreamerPreprocessingConfig,
    architecture_signature,
)
from .model import R2DreamerWorldModel
from .preprocessing import R2DreamerObservationPreprocessor

R2_UPSTREAM_COMMIT = "546e4fab8146ea4b14e1d7726bbc1a8a1d50322f"
CHECKPOINT_FORMAT = "dream-reflex-r2dreamer"
CHECKPOINT_FORMAT_VERSION = 2
FINAL_R2_CHECKPOINT_SHA256 = "5260b7c5df88929e21bd60c8686fe7b9dc43db7b2c6238cf49524b42a33ac9fc"
FINAL_R2_ARCHITECTURE_SIGNATURE = "560ed2c44b18e3fcf573160ca9ace32f26390d69797c9ea5c62f0b7cdb82e5e2"
_REQUIRED_KEYS = {
    "format",
    "format_version",
    "model_state_dict",
    "model_config",
    "preprocessing_config",
    "action_normalization",
    "step",
    "r2_upstream_commit",
    "metadata",
}


@dataclass(frozen=True)
class LoadedR2DreamerCheckpoint:
    model: R2DreamerWorldModel
    preprocessor: R2DreamerObservationPreprocessor
    action_normalizer: R2DreamerActionNormalizer
    metadata: dict[str, Any]
    step: int


def checkpoint_sha256(path: Path | str) -> str:
    """Hash a checkpoint without reading it into memory as one large byte string."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_final_checkpoint_sha256(path: Path | str) -> None:
    """Fail unless *path* is byte-for-byte the final September R2 checkpoint."""
    actual = checkpoint_sha256(path)
    if actual != FINAL_R2_CHECKPOINT_SHA256:
        raise ValueError(
            "R2 checkpoint SHA-256 does not match the final latest.pt artifact: "
            f"{actual} != {FINAL_R2_CHECKPOINT_SHA256}"
        )


def save_r2dreamer_checkpoint(
    path: Path | str,
    *,
    model: R2DreamerWorldModel,
    preprocessing_config: R2DreamerPreprocessingConfig,
    action_normalizer: R2DreamerActionNormalizer,
    step: int,
    metadata: dict[str, Any] | None = None,
    optimizer_state_dict: dict[str, Any] | None = None,
    scheduler_state_dict: dict[str, Any] | None = None,
    trainer_state: dict[str, Any] | None = None,
) -> Path:
    """Write format-v2 checkpoints accepted directly by the WP2A runtime.

    ``dream-reflex-r2dreamer`` is retained as the legacy schema identifier. It
    describes checkpoint compatibility, not the current repository name.
    """
    if action_normalizer.action_dim != model.config.action_dim:
        raise ValueError("action normalization dimension does not match the model")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "model_config": model.config.to_dict(),
        "preprocessing_config": preprocessing_config.to_dict(),
        "action_normalization": action_normalizer.to_dict(),
        "step": int(step),
        "r2_upstream_commit": R2_UPSTREAM_COMMIT,
        "metadata": dict(metadata or {}),
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict
    if scheduler_state_dict is not None:
        payload["scheduler_state_dict"] = scheduler_state_dict
    if trainer_state is not None:
        payload["trainer_state"] = trainer_state
    torch.save(payload, destination)
    return destination


def read_r2dreamer_checkpoint(
    path: Path | str,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"R2 checkpoint does not exist: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise ValueError(f"could not read R2 checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("not a compatible R2-Dreamer checkpoint")
    missing = _REQUIRED_KEYS.difference(payload)
    if missing:
        raise ValueError(f"R2 checkpoint is missing required keys: {sorted(missing)}")
    if payload.get("format_version") == 1:
        raise ValueError("legacy R2 checkpoint has no frozen action normalization")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported R2 checkpoint format version {payload.get('format_version')!r}"
        )
    if payload.get("r2_upstream_commit") != R2_UPSTREAM_COMMIT:
        raise ValueError("checkpoint R2 source commit does not match this adapted subset")
    return payload


def load_r2dreamer_checkpoint(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> LoadedR2DreamerCheckpoint:
    """Load the exact checkpoint schema and all model parameters strictly."""
    target_device = torch.device(device)
    payload = read_r2dreamer_checkpoint(path, map_location=target_device)
    try:
        model_config = R2DreamerModelConfig.from_dict(payload["model_config"])
        preprocessing_config = R2DreamerPreprocessingConfig.from_dict(
            payload["preprocessing_config"]
        )
        action_normalizer = R2DreamerActionNormalizer.from_dict(payload["action_normalization"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"incompatible R2 checkpoint configuration: {exc}") from exc

    expected_channels = len(preprocessing_config.image_keys) * 3
    if model_config.image_channels != expected_channels:
        raise ValueError(
            "checkpoint image-key metadata is incompatible with model channels: "
            f"{expected_channels} != {model_config.image_channels}"
        )
    if model_config.image_size != preprocessing_config.image_size:
        raise ValueError("checkpoint model and preprocessing image sizes differ")
    if preprocessing_config.state_mean is not None and (
        len(preprocessing_config.state_mean) != model_config.state_dim
    ):
        raise ValueError("checkpoint state normalization dimension is incompatible")
    if (preprocessing_config.state_key is None) != (model_config.state_dim == 0):
        raise ValueError("checkpoint state-key metadata is incompatible with model state input")
    if action_normalizer.action_dim != model_config.action_dim:
        raise ValueError("checkpoint action normalization dimension is incompatible")

    metadata_raw = payload["metadata"]
    if not isinstance(metadata_raw, dict):
        raise ValueError("checkpoint metadata must be a mapping")
    metadata = dict(metadata_raw)
    recorded_signature = metadata.get("architecture_signature")
    actual_signature = architecture_signature(model_config)
    if recorded_signature is not None and recorded_signature != actual_signature:
        raise ValueError(
            "checkpoint architecture signature is inconsistent with model_config: "
            f"{recorded_signature} != {actual_signature}"
        )
    metadata_normalization = metadata.get("action_normalization")
    if (
        metadata_normalization is not None
        and R2DreamerActionNormalizer.from_dict(metadata_normalization) != action_normalizer
    ):
        raise ValueError("checkpoint action normalization metadata is inconsistent")
    action_names = metadata.get("action_metadata", {}).get("names")
    if action_names is not None and tuple(action_names) != action_normalizer.action_names:
        raise ValueError("checkpoint action metadata and normalization order differ")

    model = R2DreamerWorldModel(model_config).to(target_device)
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"checkpoint model weights are incompatible: {exc}") from exc
    model.eval()
    return LoadedR2DreamerCheckpoint(
        model=model,
        preprocessor=R2DreamerObservationPreprocessor(preprocessing_config, device=target_device),
        action_normalizer=action_normalizer,
        metadata=metadata,
        step=int(payload["step"]),
    )
