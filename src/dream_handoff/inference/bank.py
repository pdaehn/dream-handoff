"""Generated and activation-anchored candidate-bank records."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _validate_actions(name: str, value: Tensor) -> None:
    if not torch.is_tensor(value) or value.ndim != 3:
        raise ValueError(f"{name} must be a tensor with shape [N,H,A]")
    if any(size <= 0 for size in value.shape):
        raise ValueError(f"{name} dimensions must be positive")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or Inf")


@dataclass(frozen=True)
class GeneratedCandidateActions:
    """Full request-time policy/control chunks before activation-time dreaming."""

    policy_actions: Tensor
    control_actions: Tensor
    request_action_count: int
    sampled_noise: Tensor | None = None

    def __post_init__(self) -> None:
        _validate_actions("policy_actions", self.policy_actions)
        _validate_actions("control_actions", self.control_actions)
        if self.policy_actions.shape[:2] != self.control_actions.shape[:2]:
            raise ValueError("policy and control actions must align on [N,H]")
        if self.policy_actions.device != self.control_actions.device:
            raise ValueError("policy and control actions must use the same device")
        if self.request_action_count < 0:
            raise ValueError("request_action_count must be non-negative")
        if self.sampled_noise is not None:
            noise = self.sampled_noise
            if not torch.is_tensor(noise) or noise.ndim != 3:
                raise ValueError("sampled_noise must have shape [N,H,D]")
            if noise.shape[:2] != self.policy_actions.shape[:2] or noise.shape[-1] <= 0:
                raise ValueError("sampled_noise must align with candidate actions on [N,H]")
            if noise.device != self.policy_actions.device or not noise.is_floating_point():
                raise ValueError("sampled_noise must be floating point on the policy device")
            if not bool(torch.isfinite(noise).all()):
                raise ValueError("sampled_noise contains NaN or Inf")

    @property
    def num_candidates(self) -> int:
        return int(self.policy_actions.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.policy_actions.shape[1])


@dataclass(frozen=True)
class CandidateBank:
    """One cropped bank dreamed from its fresh activation-time R2 posterior."""

    policy_actions: Tensor
    control_actions: Tensor
    dreamed_matching_features: Tensor
    origin_action_count: int

    def __post_init__(self) -> None:
        _validate_actions("policy_actions", self.policy_actions)
        _validate_actions("control_actions", self.control_actions)
        if self.policy_actions.shape[:2] != self.control_actions.shape[:2]:
            raise ValueError("policy and control actions must align on [N,H]")
        if self.policy_actions.device != self.control_actions.device:
            raise ValueError("policy and control actions must use the same device")
        features = self.dreamed_matching_features
        if not torch.is_tensor(features) or features.ndim != 3:
            raise ValueError("dreamed_matching_features must have shape [N,H,F]")
        if features.shape[:2] != self.policy_actions.shape[:2]:
            raise ValueError("dreamed matching features must align with candidate [N,H]")
        if features.device != self.control_actions.device:
            raise ValueError(
                "dreamed matching features and control actions must use the same device"
            )
        if features.shape[-1] <= 0 or not features.is_floating_point():
            raise ValueError("dreamed matching features require a floating feature dimension")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("dreamed_matching_features contains NaN or Inf")
        if self.origin_action_count < 0:
            raise ValueError("origin_action_count must be non-negative")

    @property
    def num_candidates(self) -> int:
        return int(self.policy_actions.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.policy_actions.shape[1])
