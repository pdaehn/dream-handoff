"""LaProp and adaptive gradient clipping used by the final R2 run.

LaProp portions are distributed under this MIT notice:

Copyright (c) 2020 Wang, T. Zhikang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


class LaProp(Optimizer):
    """Pinned R2-Dreamer LaProp update equations (Wang, 2020, MIT)."""

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 4e-5,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-20,
        weight_decay: float = 0.0,
    ) -> None:
        self.steps_before_using_centered = 10
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "amsgrad": False,
                "centered": False,
            },
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("LaProp does not support sparse gradients")
                state: dict[str, Any] = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_lr_1"] = 0.0
                    state["exp_avg_lr_2"] = 0.0
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                beta1, beta2 = group["betas"]
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                state["exp_avg_lr_1"] = state["exp_avg_lr_1"] * beta1 + (1 - beta1) * group["lr"]
                state["exp_avg_lr_2"] = state["exp_avg_lr_2"] * beta2 + (1 - beta2)
                correction = state["exp_avg_lr_1"] / group["lr"] if group["lr"] else 1.0
                denominator = exp_avg_sq.div(state["exp_avg_lr_2"]).sqrt_().add_(group["eps"])
                normalized_gradient = gradient / denominator
                exp_avg.mul_(beta1).add_(normalized_gradient, alpha=(1 - beta1) * group["lr"])
                parameter.add_(exp_avg, alpha=-1 / correction)
                if group["weight_decay"]:
                    parameter.add_(parameter, alpha=-group["weight_decay"])
        return loss


@torch.no_grad()
def clip_grad_agc_(parameters: Iterable[Tensor], *, clip: float = 0.3, pmin: float = 1e-3) -> None:
    """Clip every gradient relative to its complete parameter tensor norm."""
    pairs = [(parameter, parameter.grad) for parameter in parameters if parameter.grad is not None]
    groups: dict[tuple[torch.device, torch.dtype], list[tuple[Tensor, Tensor]]] = {}
    for parameter, gradient in pairs:
        groups.setdefault((parameter.device, parameter.dtype), []).append((parameter, gradient))
    for values in groups.values():
        parameters_group = [parameter for parameter, _ in values]
        gradients = [gradient for _, gradient in values]
        parameter_norms = torch._foreach_norm(parameters_group, ord=2)
        gradient_norms = torch._foreach_norm(gradients, ord=2)
        upper = torch._foreach_mul(torch._foreach_maximum(parameter_norms, pmin), clip)
        scales = torch._foreach_reciprocal(
            torch._foreach_maximum(torch._foreach_div(gradient_norms, upper), 1.0)
        )
        torch._foreach_mul_(gradients, scales)
