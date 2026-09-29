"""Categorical recurrent state-space model adapted from R2-Dreamer.

Upstream commit: 546e4fab8146ea4b14e1d7726bbc1a8a1d50322f (MIT).
Training uses straight-through Gumbel samples. Runtime calls remain categorical-mode
deterministic by default.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import R2DreamerModelConfig
from .distributions import OneHotDist, categorical_kl
from .networks import BlockLinear, LambdaLayer, weight_init_


def _right_pad(value: torch.Tensor, dimensions: int) -> torch.Tensor:
    for _ in range(dimensions):
        value = value.unsqueeze(-1)
    return value


class DeterministicTransition(nn.Module):
    """Upstream block-GRU deterministic transition."""

    def __init__(self, config: R2DreamerModelConfig) -> None:
        super().__init__()
        activation = getattr(nn, config.activation)
        self.blocks = config.blocks
        self.deter = config.deter
        self.deter_in = nn.Sequential(
            nn.Linear(config.deter, config.hidden),
            nn.RMSNorm(config.hidden, eps=1e-4, dtype=torch.float32),
            activation(),
        )
        self.stoch_in = nn.Sequential(
            nn.Linear(config.stoch * config.discrete, config.hidden),
            nn.RMSNorm(config.hidden, eps=1e-4, dtype=torch.float32),
            activation(),
        )
        self.action_in = nn.Sequential(
            nn.Linear(config.action_dim, config.hidden),
            nn.RMSNorm(config.hidden, eps=1e-4, dtype=torch.float32),
            activation(),
        )
        input_dim = (3 * config.hidden + config.deter // config.blocks) * config.blocks
        hidden: list[nn.Module] = []
        for _ in range(config.dyn_layers):
            hidden.extend(
                [
                    BlockLinear(input_dim, config.deter, config.blocks),
                    nn.RMSNorm(config.deter, eps=1e-4, dtype=torch.float32),
                    activation(),
                ]
            )
            input_dim = config.deter
        self.hidden = nn.Sequential(*hidden)
        self.gru = BlockLinear(input_dim, 3 * config.deter, config.blocks)

    def _flat_to_group(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*value.shape[:-1], self.blocks, -1)

    @staticmethod
    def _group_to_flat(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*value.shape[:-2], -1)

    def forward(
        self,
        stoch: torch.Tensor,
        deter: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = action.shape[0]
        stoch = stoch.reshape(batch_size, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        inputs = torch.cat(
            [self.deter_in(deter), self.stoch_in(stoch), self.action_in(action)], dim=-1
        )
        inputs = inputs.unsqueeze(-2).expand(-1, self.blocks, -1)
        inputs = self._group_to_flat(torch.cat([self._flat_to_group(deter), inputs], dim=-1))
        gates = self.gru(self.hidden(inputs))
        reset, candidate, update = (
            self._group_to_flat(value)
            for value in torch.chunk(self._flat_to_group(gates), 3, dim=-1)
        )
        reset = torch.sigmoid(reset)
        candidate = torch.tanh(reset * candidate)
        update = torch.sigmoid(update - 1)
        return update * candidate + (1 - update) * deter


class RSSM(nn.Module):
    def __init__(self, config: R2DreamerModelConfig, embed_size: int) -> None:
        super().__init__()
        self.config = config
        self.flat_stoch = config.stoch * config.discrete
        self.feat_size = self.flat_stoch + config.deter
        self.deter_transition = DeterministicTransition(config)
        activation = getattr(nn, config.activation)

        observation_layers: list[nn.Module] = []
        input_dim = config.deter + embed_size
        for _ in range(config.obs_layers):
            observation_layers.extend(
                [
                    nn.Linear(input_dim, config.hidden),
                    nn.RMSNorm(config.hidden, eps=1e-4, dtype=torch.float32),
                    activation(),
                ]
            )
            input_dim = config.hidden
        observation_layers.extend(
            [
                nn.Linear(input_dim, self.flat_stoch),
                LambdaLayer(
                    lambda value: value.reshape(*value.shape[:-1], config.stoch, config.discrete)
                ),
            ]
        )
        self.observation_net = nn.Sequential(*observation_layers)

        imagination_layers: list[nn.Module] = []
        input_dim = config.deter
        for _ in range(config.img_layers):
            imagination_layers.extend(
                [
                    nn.Linear(input_dim, config.hidden),
                    nn.RMSNorm(config.hidden, eps=1e-4, dtype=torch.float32),
                    activation(),
                ]
            )
            input_dim = config.hidden
        imagination_layers.extend(
            [
                nn.Linear(input_dim, self.flat_stoch),
                LambdaLayer(
                    lambda value: value.reshape(*value.shape[:-1], config.stoch, config.discrete)
                ),
            ]
        )
        self.imagination_net = nn.Sequential(*imagination_layers)
        self.apply(weight_init_)

    def initial(
        self, batch_size: int, *, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
        stoch = torch.zeros(
            batch_size,
            self.config.stoch,
            self.config.discrete,
            dtype=torch.float32,
            device=device,
        )
        deter = torch.zeros(
            batch_size,
            self.config.deter,
            dtype=torch.float32,
            device=device,
        )
        return stoch, deter

    def get_dist(self, logits: torch.Tensor) -> OneHotDist:
        return OneHotDist(logits, unimix_ratio=self.config.unimix_ratio)

    def _stochastic_state(self, logits: torch.Tensor, deterministic: bool) -> torch.Tensor:
        distribution = self.get_dist(logits)
        return distribution.mode if deterministic else distribution.rsample()

    def obs_step(
        self,
        stoch: torch.Tensor,
        deter: torch.Tensor,
        previous_action: torch.Tensor,
        embed: torch.Tensor,
        reset: torch.Tensor,
        *,
        deterministic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reset = reset.bool()
        stoch = torch.where(
            _right_pad(reset, stoch.ndim - reset.ndim), torch.zeros_like(stoch), stoch
        )
        deter = torch.where(
            _right_pad(reset, deter.ndim - reset.ndim), torch.zeros_like(deter), deter
        )
        previous_action = torch.where(
            _right_pad(reset, previous_action.ndim - reset.ndim),
            torch.zeros_like(previous_action),
            previous_action,
        )
        deter = self.deter_transition(stoch, deter, previous_action)
        logits = self.observation_net(torch.cat([deter, embed], dim=-1))
        return self._stochastic_state(logits, deterministic), deter, logits

    def observe(
        self,
        embed: torch.Tensor,
        previous_actions: torch.Tensor,
        initial: tuple[torch.Tensor, torch.Tensor],
        reset: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stoch, deter = initial
        stochastic_states, deterministic_states, logits = [], [], []
        for index in range(previous_actions.shape[1]):
            stoch, deter, logit = self.obs_step(
                stoch,
                deter,
                previous_actions[:, index],
                embed[:, index],
                reset[:, index],
                deterministic=deterministic,
            )
            stochastic_states.append(stoch)
            deterministic_states.append(deter)
            logits.append(logit)
        return (
            torch.stack(stochastic_states, dim=1),
            torch.stack(deterministic_states, dim=1),
            torch.stack(logits, dim=1),
        )

    def prior_logits(self, deter: torch.Tensor) -> torch.Tensor:
        return self.imagination_net(deter)

    def prior(self, deter: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        return self._stochastic_state(self.prior_logits(deter), deterministic)

    def img_step(
        self,
        stoch: torch.Tensor,
        deter: torch.Tensor,
        previous_action: torch.Tensor,
        *,
        deterministic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        deter = self.deter_transition(stoch, deter, previous_action)
        return self.prior(deter, deterministic=deterministic), deter

    def imagine_with_action(
        self,
        stoch: torch.Tensor,
        deter: torch.Tensor,
        actions: torch.Tensor,
        *,
        deterministic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stochastic_states, deterministic_states = [], []
        for index in range(actions.shape[1]):
            stoch, deter = self.img_step(
                stoch, deter, actions[:, index], deterministic=deterministic
            )
            stochastic_states.append(stoch)
            deterministic_states.append(deter)
        return torch.stack(stochastic_states, dim=1), torch.stack(deterministic_states, dim=1)

    def get_feat(self, stoch: torch.Tensor, deter: torch.Tensor) -> torch.Tensor:
        stoch = stoch.reshape(*stoch.shape[:-2], self.flat_stoch)
        return torch.cat([stoch, deter], dim=-1)

    @staticmethod
    def kl_loss(
        posterior_logits: torch.Tensor,
        prior_logits: torch.Tensor,
        free_bits: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representation = categorical_kl(posterior_logits, prior_logits.detach()).sum(-1)
        dynamics = categorical_kl(posterior_logits.detach(), prior_logits).sum(-1)
        return (
            torch.clip(dynamics, min=free_bits),
            torch.clip(representation, min=free_bits),
        )
