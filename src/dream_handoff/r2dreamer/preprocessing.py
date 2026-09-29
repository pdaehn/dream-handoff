"""Checkpoint-coupled R2 observation preprocessing for live inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from .config import R2DreamerPreprocessingConfig


def prepare_image_tensor(
    raw_image: Any,
    *,
    key: str,
    image_size: tuple[int, int],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert one CHW/HWC image to float32 channel-last form and resize it."""
    image = torch.as_tensor(raw_image, device=device)
    if image.ndim < 3:
        raise ValueError(f"image {key!r} must have at least three dimensions")
    if image.dtype == torch.uint8:
        image = image.to(dtype=torch.float32) / 255.0
    else:
        image = image.to(dtype=torch.float32)
        if not bool(torch.isfinite(image).all()):
            raise ValueError(f"image {key!r} contains NaN or Inf")
        if image.numel() and (image.min() < 0 or image.max() > 1):
            raise ValueError(f"floating image {key!r} must already use [0,1] semantics")

    if image.shape[-3] in (1, 3, 4):
        channel_first = image
    elif image.shape[-1] in (1, 3, 4):
        channel_first = image.movedim(-1, -3)
    else:
        raise ValueError(
            f"cannot identify channel dimension for image {key!r}: {tuple(image.shape)}"
        )

    leading = channel_first.shape[:-3]
    flat = channel_first.reshape(-1, *channel_first.shape[-3:])
    if tuple(flat.shape[-2:]) != image_size:
        flat = F.interpolate(flat, size=image_size, mode="bilinear", align_corners=False)
    channel_last = flat.permute(0, 2, 3, 1)
    return channel_last.reshape(*leading, *channel_last.shape[-3:])


class R2DreamerObservationPreprocessor:
    """Select ordered cameras and optional proprioception for the R2 encoder."""

    def __init__(
        self,
        config: R2DreamerPreprocessingConfig,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config
        self.device = None if device is None else torch.device(device)

    def reset(self) -> None:
        """Reset the preprocessing boundary, which is currently stateless."""

    def __call__(self, observation: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        if self.config.image_keys:
            images = [self._prepare_image(observation, key) for key in self.config.image_keys]
            leading = images[0].shape[:-3]
            if any(image.shape[:-3] != leading for image in images[1:]):
                raise ValueError("configured camera images must have matching leading dimensions")
            result["images"] = torch.cat(images, dim=-1)

        if self.config.state_key is not None:
            try:
                raw_state = observation[self.config.state_key]
            except KeyError as exc:
                raise ValueError(
                    f"R2 observation is missing state key {self.config.state_key!r}"
                ) from exc
            state = torch.as_tensor(raw_state, device=self.device).to(dtype=torch.float32)
            if state.ndim < 1:
                raise ValueError("R2 vector state must have a feature dimension")
            if not bool(torch.isfinite(state).all()):
                raise ValueError("R2 vector state contains NaN or Inf")
            if self.config.state_mean is not None:
                mean = state.new_tensor(self.config.state_mean)
                std = state.new_tensor(self.config.state_std)
                if state.shape[-1] != mean.numel():
                    raise ValueError(
                        "state normalization dimension does not match observation: "
                        f"{mean.numel()} != {state.shape[-1]}"
                    )
                state = (state - mean) / std
            result["state"] = state
        return result

    def _prepare_image(self, observation: Mapping[str, Any], key: str) -> torch.Tensor:
        try:
            raw_image = observation[key]
        except KeyError as exc:
            raise ValueError(f"R2 observation is missing image key {key!r}") from exc
        image = prepare_image_tensor(
            raw_image,
            key=key,
            image_size=self.config.image_size,
            device=self.device,
        )
        if image.shape[-1] != 3:
            raise ValueError(f"R2 image {key!r} must have exactly three RGB channels")
        return image
