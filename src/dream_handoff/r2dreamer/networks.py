"""R2-Dreamer encoder and network primitives retained for checkpoint inference.

Adapted from NM512/r2dreamer commit
546e4fab8146ea4b14e1d7726bbc1a8a1d50322f under the MIT license.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .config import R2DreamerModelConfig


def weight_init_(module: nn.Module, fan_type: str = "in") -> None:
    """Apply the upstream truncated-normal initialization."""
    if isinstance(module, nn.RMSNorm):
        with torch.no_grad():
            module.weight.fill_(1.0)
        return
    weight = getattr(module, "weight", None)
    if weight is None or weight.numel() == 0 or weight.ndim < 2:
        return
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
    fan = {"avg": (fan_in + fan_out) / 2, "in": fan_in, "out": fan_out}[fan_type]
    std = 1.1368 * math.sqrt(1 / fan)
    with torch.no_grad():
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        bias = getattr(module, "bias", None)
        if bias is not None:
            bias.zero_()


def symlog(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.log1p(torch.abs(value))


class LambdaLayer(nn.Module):
    def __init__(self, function: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.function = function

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.function(value)


class BlockLinear(nn.Module):
    """Independent linear projections over deterministic-state blocks."""

    def __init__(self, in_channels: int, out_channels: int, blocks: int) -> None:
        super().__init__()
        if in_channels % blocks or out_channels % blocks:
            raise ValueError("BlockLinear dimensions must be divisible by blocks")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks = blocks
        self.weight = nn.Parameter(
            torch.empty(out_channels // blocks, in_channels // blocks, blocks)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        leading = value.shape[:-1]
        value = value.view(*leading, self.blocks, self.in_channels // self.blocks)
        value = torch.einsum("...gi,oig->...go", value, self.weight)
        return value.reshape(*leading, self.out_channels) + self.bias


class Conv2dSamePad(nn.Conv2d):
    @staticmethod
    def _same_pad(size: int, kernel: int, stride: int, dilation: int) -> int:
        output = (size + stride - 1) // stride
        return max((output - 1) * stride + (kernel - 1) * dilation + 1 - size, 0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        pad_h = self._same_pad(height, self.kernel_size[0], self.stride[0], self.dilation[0])
        pad_w = self._same_pad(width, self.kernel_size[1], self.stride[1], self.dilation[1])
        if pad_h or pad_w:
            value = F.pad(
                value,
                [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
            )
        return F.conv2d(
            value,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class RMSNorm2D(nn.RMSNorm):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return super().forward(value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvEncoder(nn.Module):
    """Channel-last convolutional encoder used by the final checkpoint."""

    def __init__(self, config: R2DreamerModelConfig) -> None:
        super().__init__()
        activation = getattr(nn, config.activation)
        height, width = config.image_size
        depths = tuple(config.encoder_depth * multiplier for multiplier in config.encoder_mults)
        layers: list[nn.Module] = []
        in_channels = config.image_channels
        for depth in depths:
            layers.extend(
                [
                    Conv2dSamePad(
                        in_channels,
                        depth,
                        config.encoder_kernel_size,
                        stride=1,
                        bias=True,
                    ),
                    nn.MaxPool2d(2, 2),
                    RMSNorm2D(depth, eps=1e-4, dtype=torch.float32),
                    activation(),
                ]
            )
            in_channels = depth
            height //= 2
            width //= 2

        self.out_dim = depths[-1] * height * width
        self.layers = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        leading = observation.shape[:-3]
        value = (observation - 0.5).reshape(-1, *observation.shape[-3:])
        value = self.layers(value.permute(0, 3, 1, 2))
        return value.reshape(*leading, -1)


class VectorEncoder(nn.Module):
    def __init__(self, config: R2DreamerModelConfig) -> None:
        super().__init__()
        activation = getattr(nn, config.activation)
        layers: list[nn.Module] = []
        input_dim = config.state_dim
        for _ in range(config.vector_layers):
            layers.extend(
                [
                    nn.Linear(input_dim, config.vector_units),
                    nn.RMSNorm(config.vector_units, eps=1e-4, dtype=torch.float32),
                    activation(),
                ]
            )
            input_dim = config.vector_units
        self.out_dim = config.vector_units
        self.layers = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(symlog(observation))


class R2DreamerEncoder(nn.Module):
    """Compose the ordered camera and vector encoders used at training time."""

    def __init__(self, config: R2DreamerModelConfig) -> None:
        super().__init__()
        self.image_encoder = ConvEncoder(config) if config.image_channels else None
        self.vector_encoder = VectorEncoder(config) if config.state_dim else None
        self.out_dim = sum(
            encoder.out_dim
            for encoder in (self.image_encoder, self.vector_encoder)
            if encoder is not None
        )
        self.apply(weight_init_)

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded: list[torch.Tensor] = []
        if self.image_encoder is not None:
            encoded.append(self.image_encoder(observation["images"]))
        if self.vector_encoder is not None:
            encoded.append(self.vector_encoder(observation["state"]))
        return encoded[0] if len(encoded) == 1 else torch.cat(encoded, dim=-1)


class Projector(nn.Module):
    """Training-time projection retained because it is present in the checkpoint."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(input_dim, output_dim, bias=False)
        self.apply(weight_init_)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.w(value)
