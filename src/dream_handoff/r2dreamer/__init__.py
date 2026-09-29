"""Concrete adapted R2-Dreamer runtime used by DreamHandoff."""

from .action_normalization import R2DreamerActionNormalizer
from .checkpoint import (
    FINAL_R2_ARCHITECTURE_SIGNATURE,
    FINAL_R2_CHECKPOINT_SHA256,
    checkpoint_sha256,
    load_r2dreamer_checkpoint,
    save_r2dreamer_checkpoint,
    verify_final_checkpoint_sha256,
)
from .config import (
    R2DreamerLossConfig,
    R2DreamerModelConfig,
    R2DreamerPreprocessingConfig,
    R2DreamerTrainingConfig,
)
from .model import R2State, R2Trajectory
from .preprocessing import R2DreamerObservationPreprocessor, prepare_image_tensor
from .runtime import R2DreamerRuntime

__all__ = [
    "FINAL_R2_ARCHITECTURE_SIGNATURE",
    "FINAL_R2_CHECKPOINT_SHA256",
    "R2DreamerActionNormalizer",
    "R2DreamerLossConfig",
    "R2DreamerModelConfig",
    "R2DreamerObservationPreprocessor",
    "R2DreamerPreprocessingConfig",
    "R2DreamerRuntime",
    "R2DreamerTrainingConfig",
    "R2State",
    "R2Trajectory",
    "checkpoint_sha256",
    "load_r2dreamer_checkpoint",
    "prepare_image_tensor",
    "save_r2dreamer_checkpoint",
    "verify_final_checkpoint_sha256",
]
