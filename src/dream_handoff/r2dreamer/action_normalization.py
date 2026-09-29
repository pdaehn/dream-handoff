"""Immutable canonical-action normalization used by the final R2 checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class R2DreamerActionNormalizer:
    """Map ordered raw robot actions with frozen train-split q01/q99 statistics."""

    action_names: tuple[str, ...]
    q01: tuple[float, ...]
    q99: tuple[float, ...]
    epsilon: float = 1e-8
    mode: str = "quantiles"

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_names", tuple(self.action_names))
        object.__setattr__(self, "q01", tuple(float(value) for value in self.q01))
        object.__setattr__(self, "q99", tuple(float(value) for value in self.q99))
        if self.mode != "quantiles":
            raise ValueError("R2 action normalization mode must be 'quantiles'")
        if not self.action_names:
            raise ValueError("action_names must not be empty")
        if len(set(self.action_names)) != len(self.action_names):
            raise ValueError("action_names must be unique and ordered")
        if len(self.q01) != len(self.action_names) or len(self.q99) != len(self.action_names):
            raise ValueError("action normalization statistics must match action_names")
        if self.epsilon <= 0:
            raise ValueError("action normalization epsilon must be positive")
        lower = torch.tensor(self.q01, dtype=torch.float64)
        upper = torch.tensor(self.q99, dtype=torch.float64)
        if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
            raise ValueError("action normalization statistics must be finite")
        if bool((upper < lower).any()):
            raise ValueError("q99 must be greater than or equal to q01")

    @property
    def action_dim(self) -> int:
        return len(self.action_names)

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        """Map q01 to -1 and q99 to 1 without clipping out-of-range actions."""
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions)
        if actions.ndim == 0 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                "canonical action dimension does not match checkpoint statistics: "
                f"{tuple(actions.shape)} vs {self.action_dim}"
            )
        if not actions.is_floating_point():
            actions = actions.to(dtype=torch.float32)
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("canonical actions contain NaN or Inf")
        lower = actions.new_tensor(self.q01)
        upper = actions.new_tensor(self.q99)
        denominator = upper - lower
        denominator = torch.where(
            denominator == 0,
            denominator.new_full((), self.epsilon),
            denominator,
        )
        return 2.0 * (actions - lower) / denominator - 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> R2DreamerActionNormalizer:
        data = dict(value)
        for key in ("action_names", "q01", "q99"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)
