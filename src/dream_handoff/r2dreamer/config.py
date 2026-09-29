"""Checkpoint-coupled configuration for the retained R2-Dreamer runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class R2DreamerPreprocessingConfig:
    """Observation tensor semantics serialized in an R2 checkpoint."""

    image_keys: tuple[str, ...] = (
        "observation.images.context",
        "observation.images.wrist",
    )
    image_size: tuple[int, int] = (64, 64)
    state_key: str | None = "observation.state"
    state_mean: tuple[float, ...] | None = None
    state_std: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_keys", tuple(self.image_keys))
        object.__setattr__(self, "image_size", tuple(self.image_size))
        if not self.image_keys and self.state_key is None:
            raise ValueError("R2 preprocessing requires an image or vector input")
        if len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError("image_keys must be unique and ordered")
        if len(self.image_size) != 2 or any(value <= 0 for value in self.image_size):
            raise ValueError(f"image_size must contain two positive values, got {self.image_size}")
        if (self.state_mean is None) != (self.state_std is None):
            raise ValueError("state_mean and state_std must be configured together")
        if self.state_mean is not None:
            object.__setattr__(self, "state_mean", tuple(self.state_mean))
            object.__setattr__(self, "state_std", tuple(self.state_std or ()))
            if len(self.state_mean) != len(self.state_std or ()):
                raise ValueError("state_mean and state_std lengths must match")
            if any(value <= 0 for value in self.state_std or ()):
                raise ValueError("state_std values must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> R2DreamerPreprocessingConfig:
        data = dict(value)
        for key in ("image_keys", "image_size", "state_mean", "state_std"):
            if data.get(key) is not None:
                data[key] = tuple(data[key])
        return cls(**data)


@dataclass(frozen=True)
class R2DreamerModelConfig:
    """Learned dimensions needed to reconstruct the checkpoint model exactly."""

    action_dim: int
    image_channels: int
    state_dim: int
    image_size: tuple[int, int] = (64, 64)
    stoch: int = 32
    deter: int = 2048
    hidden: int = 256
    discrete: int = 16
    blocks: int = 8
    obs_layers: int = 1
    img_layers: int = 2
    dyn_layers: int = 1
    unimix_ratio: float = 0.01
    encoder_depth: int = 16
    encoder_mults: tuple[int, ...] = (2, 3, 4, 4)
    encoder_kernel_size: int = 5
    vector_layers: int = 3
    vector_units: int = 256
    activation: str = "SiLU"

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_size", tuple(self.image_size))
        object.__setattr__(self, "encoder_mults", tuple(self.encoder_mults))
        positive = {
            "action_dim": self.action_dim,
            "stoch": self.stoch,
            "deter": self.deter,
            "hidden": self.hidden,
            "discrete": self.discrete,
            "blocks": self.blocks,
            "obs_layers": self.obs_layers,
            "img_layers": self.img_layers,
            "dyn_layers": self.dyn_layers,
            "encoder_depth": self.encoder_depth,
            "encoder_kernel_size": self.encoder_kernel_size,
            "vector_layers": self.vector_layers,
            "vector_units": self.vector_units,
        }
        if self.image_channels <= 0 and self.state_dim <= 0:
            raise ValueError("R2 encoder requires image_channels or state_dim")
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.image_channels < 0 or self.state_dim < 0:
            raise ValueError("image_channels and state_dim cannot be negative")
        if self.deter % self.blocks:
            raise ValueError("deter must be divisible by blocks")
        if not 0.0 <= self.unimix_ratio < 1.0:
            raise ValueError("unimix_ratio must be in [0, 1)")
        if self.image_channels:
            factor = 2 ** len(self.encoder_mults)
            if any(size < factor or size % factor for size in self.image_size):
                raise ValueError(
                    "image_size must be divisible by encoder factor "
                    f"{factor}, got {self.image_size}"
                )
        if not hasattr(torch.nn, self.activation):
            raise ValueError(f"unknown torch activation {self.activation!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> R2DreamerModelConfig:
        data = dict(value)
        for key in ("image_size", "encoder_mults"):
            if data.get(key) is not None:
                data[key] = tuple(data[key])
        return cls(**data)


@dataclass(frozen=True)
class R2DreamerLossConfig:
    """The active loss constants from the final s12-r64 run."""

    dynamics_scale: float = 1.0
    representation_scale: float = 0.1
    barlow_scale: float = 0.05
    free_bits: float = 1.0
    barlow_lambda: float = 5e-4

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative, got {value}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class R2DreamerTrainingConfig:
    """Concrete offline training settings for the final s12-r64 model."""

    dataset_root: Path
    split_path: Path
    output_dir: Path
    device: str = "cuda"
    replay_storage_device: str = "cuda"
    seed: int = 0
    steps: int = 25_000
    sequence_length: int = 64
    batch_size: int = 16
    num_workers: int = 4
    validation_interval: int = 500
    checkpoint_interval: int = 500
    validation_batches: int = 20
    learning_rate: float = 4e-5
    warmup_steps: int = 1000
    cache_dir: Path | None = None
    cache_preprocessed_dataset: bool = True
    resume_path: Path | None = None
    video_backend: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "steps",
            "sequence_length",
            "batch_size",
            "validation_interval",
            "checkpoint_interval",
            "validation_batches",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if self.batch_size * self.sequence_length < 2:
            raise ValueError("R2 Barlow training requires at least two batch/time samples")
        if self.video_backend not in {None, "torchcodec", "pyav"}:
            raise ValueError("video_backend must be 'torchcodec', 'pyav', or None")
        for name in ("dataset_root", "split_path", "output_dir", "cache_dir", "resume_path"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for name in ("dataset_root", "split_path", "output_dir", "cache_dir", "resume_path"):
            value = data[name]
            data[name] = None if value is None else str(value)
        return data


_ARCHITECTURE_SIGNATURE_FIELDS = (
    "stoch",
    "deter",
    "hidden",
    "discrete",
    "blocks",
    "obs_layers",
    "img_layers",
    "dyn_layers",
    "encoder_depth",
    "encoder_mults",
    "encoder_kernel_size",
    "vector_layers",
    "vector_units",
    "activation",
    "image_size",
    "image_channels",
    "state_dim",
)


def architecture_signature(config: R2DreamerModelConfig) -> str:
    """Return the frozen implementation's shape-defining config fingerprint."""
    payload = {field: getattr(config, field) for field in _ARCHITECTURE_SIGNATURE_FIELDS}
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
