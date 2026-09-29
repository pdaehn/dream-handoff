"""Concrete state records and learned modules for R2 inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import R2DreamerModelConfig
from .networks import Projector, R2DreamerEncoder
from .rssm import RSSM


def _validate_parts(stoch: torch.Tensor, deter: torch.Tensor, *, name: str) -> None:
    if not torch.is_tensor(stoch) or not torch.is_tensor(deter):
        raise ValueError(f"{name}.stoch and {name}.deter must be tensors")
    if stoch.ndim != deter.ndim + 1 or stoch.shape[:-2] != deter.shape[:-1]:
        raise ValueError(
            f"{name} categorical and deterministic leading dimensions differ: "
            f"{tuple(stoch.shape)} and {tuple(deter.shape)}"
        )
    if any(size <= 0 for size in stoch.shape[-2:]) or deter.shape[-1] <= 0:
        raise ValueError(f"{name} state dimensions must be positive")


@dataclass(frozen=True)
class R2State:
    """One observation-conditioned categorical RSSM posterior state."""

    stoch: torch.Tensor
    deter: torch.Tensor

    def __post_init__(self) -> None:
        _validate_parts(self.stoch, self.deter, name="R2State")

    def detached_clone(self) -> R2State:
        return R2State(self.stoch.detach().clone(), self.deter.detach().clone())


@dataclass(frozen=True)
class R2Trajectory:
    """Pre-action-aligned candidate states with leading dimensions ``[N,H]``."""

    stoch: torch.Tensor
    deter: torch.Tensor

    def __post_init__(self) -> None:
        _validate_parts(self.stoch, self.deter, name="R2Trajectory")
        if self.stoch.ndim != 4 or self.deter.ndim != 3:
            raise ValueError("R2Trajectory must have shapes stoch=[N,H,S,C], deter=[N,H,D]")

    @property
    def leading_shape(self) -> tuple[int, int]:
        return int(self.deter.shape[0]), int(self.deter.shape[1])


@dataclass(frozen=True)
class R2DreamerModelOutput:
    """Temporally aligned tensors used by the final R2 training objective."""

    embed: torch.Tensor
    posterior_stoch: torch.Tensor
    posterior_deter: torch.Tensor
    posterior_logits: torch.Tensor
    prior_logits: torch.Tensor
    features: torch.Tensor


class R2DreamerWorldModel(nn.Module):
    """Encoder, categorical RSSM, and checkpoint-resident R2 projector."""

    def __init__(self, config: R2DreamerModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = R2DreamerEncoder(config)
        self.rssm = RSSM(config, self.encoder.out_dim)
        self.projector = Projector(self.rssm.feat_size, self.encoder.out_dim)

    def forward(
        self,
        observation: dict[str, torch.Tensor],
        previous_actions: torch.Tensor,
        is_first: torch.Tensor,
        *,
        initial_state: R2State | None = None,
        deterministic: bool = False,
    ) -> R2DreamerModelOutput:
        """Encode ``o_t`` and update the posterior with ``a_(t-1)``."""
        if previous_actions.ndim != 3:
            raise ValueError("previous_actions must have shape [B,T,A]")
        if previous_actions.shape[-1] != self.config.action_dim:
            raise ValueError("previous_actions do not match the R2 action dimension")
        if tuple(is_first.shape) != tuple(previous_actions.shape[:2]):
            raise ValueError("is_first must have shape [B,T]")
        embed = self.encoder(observation)
        if tuple(embed.shape[:2]) != tuple(previous_actions.shape[:2]):
            raise ValueError("encoded observations and actions must share [B,T]")
        if initial_state is None:
            initial = self.rssm.initial(previous_actions.shape[0], device=previous_actions.device)
        else:
            expected_stoch = (
                previous_actions.shape[0],
                self.config.stoch,
                self.config.discrete,
            )
            expected_deter = (previous_actions.shape[0], self.config.deter)
            if tuple(initial_state.stoch.shape) != expected_stoch:
                raise ValueError(f"initial_state.stoch must have shape {expected_stoch}")
            if tuple(initial_state.deter.shape) != expected_deter:
                raise ValueError(f"initial_state.deter must have shape {expected_deter}")
            if (
                initial_state.stoch.device != previous_actions.device
                or initial_state.deter.device != previous_actions.device
            ):
                raise ValueError("initial_state and previous_actions must use the same device")
            initial = initial_state.stoch.detach(), initial_state.deter.detach()
        posterior_stoch, posterior_deter, posterior_logits = self.rssm.observe(
            embed,
            previous_actions,
            initial,
            is_first,
            deterministic=deterministic,
        )
        prior_logits = self.rssm.prior_logits(posterior_deter)
        return R2DreamerModelOutput(
            embed=embed,
            posterior_stoch=posterior_stoch,
            posterior_deter=posterior_deter,
            posterior_logits=posterior_logits,
            prior_logits=prior_logits,
            features=self.rssm.get_feat(posterior_stoch, posterior_deter),
        )
