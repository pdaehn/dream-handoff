"""The three active losses from the final R2 s12-r64 run.

Adapted from NM512/r2dreamer commit
546e4fab8146ea4b14e1d7726bbc1a8a1d50322f under the MIT license.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import R2DreamerLossConfig
from .model import R2DreamerModelOutput, R2DreamerWorldModel


@dataclass(frozen=True)
class R2DreamerLosses:
    total: torch.Tensor
    dynamics: torch.Tensor
    representation: torch.Tensor
    barlow: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss/total": float(self.total.detach()),
            "loss/dyn": float(self.dynamics.detach()),
            "loss/rep": float(self.representation.detach()),
            "loss/barlow": float(self.barlow.detach()),
        }


def barlow_twins_loss(
    model: R2DreamerWorldModel,
    output: R2DreamerModelOutput,
    coefficient: float,
) -> torch.Tensor:
    """Project posterior features toward a detached observation embedding target."""
    features = output.features.reshape(-1, output.features.shape[-1])
    embeddings = output.embed.reshape(-1, output.embed.shape[-1]).detach()
    if features.shape[0] < 2:
        raise ValueError("R2 Barlow loss requires at least two batch/time samples")
    projected = model.projector(features)
    projected = (projected - projected.mean(0)) / (projected.std(0) + 1e-8)
    embeddings = (embeddings - embeddings.mean(0)) / (embeddings.std(0) + 1e-8)
    correlation = projected.T @ embeddings / projected.shape[0]
    invariance = (torch.diagonal(correlation) - 1.0).square().sum()
    off_diagonal = ~torch.eye(correlation.shape[0], dtype=torch.bool, device=correlation.device)
    redundancy = correlation[off_diagonal].square().sum()
    return invariance + coefficient * redundancy


def world_model_loss(
    model: R2DreamerWorldModel,
    output: R2DreamerModelOutput,
    config: R2DreamerLossConfig,
) -> R2DreamerLosses:
    dynamics, representation = model.rssm.kl_loss(
        output.posterior_logits, output.prior_logits, config.free_bits
    )
    dynamics = dynamics.mean()
    representation = representation.mean()
    barlow = barlow_twins_loss(model, output, config.barlow_lambda)
    total = (
        config.dynamics_scale * dynamics
        + config.representation_scale * representation
        + config.barlow_scale * barlow
    )
    return R2DreamerLosses(total, dynamics, representation, barlow)
