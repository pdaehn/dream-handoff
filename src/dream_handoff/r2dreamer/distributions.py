"""Categorical helper adapted from R2-Dreamer.

Upstream: https://github.com/NM512/r2dreamer at commit
546e4fab8146ea4b14e1d7726bbc1a8a1d50322f (MIT).
"""

from __future__ import annotations

import torch
from torch import distributions as torchd
from torch.nn import functional as F

EMPTY_SIZE = torch.Size()


class OneHotDist(torchd.OneHotCategorical):
    """Straight-through categorical distribution with optional uniform mixing."""

    def __init__(self, logits: torch.Tensor, unimix_ratio: float = 0.0) -> None:
        probs = F.softmax(logits.float(), dim=-1)
        uniform = unimix_ratio / probs.shape[-1]
        probs = probs * (1.0 - unimix_ratio) + torch.ones_like(probs) * uniform
        super().__init__(logits=torch.log(probs))

    @property
    def mode(self) -> torch.Tensor:
        mode = F.one_hot(torch.argmax(self.logits, dim=-1), self.logits.shape[-1])
        return mode.detach() + self.logits - self.logits.detach()

    def rsample(
        self, sample_shape: torch.Size = EMPTY_SIZE, temperature: float = 1.0
    ) -> torch.Tensor:
        """Draw the straight-through Gumbel sample used by R2 training."""
        if sample_shape:
            raise NotImplementedError("R2 OneHotDist does not use sample_shape")
        return F.gumbel_softmax(self.logits, tau=temperature, hard=True, dim=-1)

    def sample(self, sample_shape: torch.Size = EMPTY_SIZE) -> torch.Tensor:
        del sample_shape
        raise NotImplementedError("use rsample() for training or mode for inference")


def categorical_kl(logits_left: torch.Tensor, logits_right: torch.Tensor) -> torch.Tensor:
    """Elementwise categorical KL with the pinned upstream formulation."""
    logprob_left = torch.log_softmax(logits_left, dim=-1)
    logprob_right = torch.log_softmax(logits_right, dim=-1)
    probability = torch.softmax(logits_left, dim=-1)
    return (probability * (logprob_left - logprob_right)).sum(dim=-1)
