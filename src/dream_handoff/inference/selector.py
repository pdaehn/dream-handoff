"""Phase-aligned baseline and absolute-hysteresis candidate selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import SelectorMode


@dataclass(frozen=True)
class SelectionResult:
    index: int
    challenger_index: int
    incumbent_index: int | None
    distances: Tensor
    switched: bool
    advantage: float | None
    rule: str


class CandidateSelector:
    """Maintain only bank-local incumbent state and a rollout-wide phase-0 RNG."""

    def __init__(self, mode: SelectorMode, *, hysteresis_tau: float | None, seed: int) -> None:
        self._mode = mode
        self._tau = hysteresis_tau
        self._rng = torch.Generator().manual_seed(seed)
        self._incumbent: int | None = None

    @property
    def incumbent(self) -> int | None:
        return self._incumbent

    def reset_bank(self) -> None:
        """Clear bank-local state without rewinding the rollout-wide RNG."""
        self._incumbent = None

    def select(
        self,
        live_matching_feature: Tensor,
        dreamed_matching_features: Tensor,
        *,
        phase: int,
    ) -> SelectionResult:
        distances = matching_distances(live_matching_feature, dreamed_matching_features)
        challenger = int(torch.argmin(distances).item())
        incumbent = self._incumbent
        if phase == 0:
            selected = int(
                torch.randint(0, int(distances.shape[0]), (1,), generator=self._rng).item()
            )
            advantage = None
            rule = "seeded_uniform"
        elif self._mode == "baseline" or incumbent is None:
            selected = challenger
            advantage = None
            rule = "nearest"
        else:
            if self._tau is None:  # pragma: no cover - config validates this
                raise RuntimeError("absolute hysteresis has no threshold")
            advantage = float(distances[incumbent].item() - distances[challenger].item())
            selected = challenger if advantage > self._tau else incumbent
            rule = "absolute_hysteresis"
        switched = incumbent is not None and selected != incumbent
        self._incumbent = selected
        return SelectionResult(
            index=selected,
            challenger_index=challenger,
            incumbent_index=incumbent,
            distances=distances,
            switched=switched,
            advantage=advantage,
            rule=rule,
        )


def matching_distances(live: Tensor, dreamed: Tensor) -> Tensor:
    """Return L2 distance for each candidate; equal minima follow torch.argmin."""
    if not torch.is_tensor(live):
        raise ValueError("live matching feature must be a tensor")
    if not torch.is_tensor(dreamed) or dreamed.ndim != 2:
        raise ValueError("dreamed matching features must have shape [N,F]")
    if live.ndim == 2 and live.shape[0] == 1:
        live = live.squeeze(0)
    if live.ndim != 1 or dreamed.shape[1] != live.shape[0] or dreamed.shape[0] == 0:
        raise ValueError("live and dreamed matching-feature shapes are incompatible")
    if not live.is_floating_point() or not dreamed.is_floating_point():
        raise ValueError("matching features must use floating-point dtypes")
    if not bool(torch.isfinite(live).all()) or not bool(torch.isfinite(dreamed).all()):
        raise ValueError("matching features contain NaN or Inf")
    live = live.to(device=dreamed.device, dtype=dreamed.dtype)
    return torch.linalg.vector_norm(dreamed - live, dim=-1)
