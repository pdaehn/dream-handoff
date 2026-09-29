"""Concrete N-way sampling through pinned LeRobot SmolVLA and RTC."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import torch
from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from torch import Tensor


def _repeat_batch(batch: Mapping[str, Any], count: int) -> dict[str, Any]:
    repeated: dict[str, Any] = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            repeated[key] = value
        elif value.ndim == 0:
            repeated[key] = value.repeat(count)
        else:
            if value.shape[0] != 1:
                raise ValueError(f"preprocessed batch {key!r} must have leading size 1")
            repeated[key] = value.repeat(count, *([1] * (value.ndim - 1)))
    return repeated


def configure_rtc_prefix_guidance(policy: Any) -> RTCConfig:
    """Install LeRobot's final linear soft-guidance processor on SmolVLA."""
    config = RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
        max_guidance_weight=10.0,
        execution_horizon=int(policy.config.chunk_size),
    )
    policy.config.rtc_config = config
    initialize = getattr(policy, "init_rtc_processor", None)
    if not callable(initialize):
        raise ValueError("SmolVLA policy does not expose init_rtc_processor()")
    initialize()
    if getattr(policy, "rtc_processor", None) is None:
        raise RuntimeError("SmolVLA did not initialize its RTC processor")
    return config


class SmolVLACandidateSampler:
    """Preprocess once, make explicit flow noise, and generate one N-sized bank."""

    def __init__(self, policy: Any, preprocessor: Any, postprocessor: Any, *, device: str) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = torch.device(device)
        self.last_noise: Tensor | None = None

    def make_noise(
        self,
        observation: Mapping[str, Any],
        num_candidates: int,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        state = observation.get("observation.state")
        if not torch.is_tensor(state):
            raise ValueError("preprocessed observation must contain tensor 'observation.state'")
        return torch.randn(
            num_candidates,
            int(self.policy.config.chunk_size),
            int(self.policy.config.max_action_dim),
            dtype=torch.float32,
            device=state.device,
            generator=generator,
        )

    def sample_candidate_bank(
        self,
        prepared_observation: Mapping[str, Any],
        num_candidates: int,
        *,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
        prefix_actions: Tensor | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return separate policy-space and canonical control-space banks."""
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if noise is not None and generator is not None:
            raise ValueError("pass explicit noise or a generator, not both")

        observation = dict(prepared_observation)
        guided = prefix_actions is not None
        autocast = (
            torch.autocast(device_type=self.device.type)
            if self.device.type == "cuda" and bool(getattr(self.policy.config, "use_amp", False))
            else nullcontext()
        )
        gradient_context = torch.no_grad() if guided else torch.inference_mode()
        with gradient_context, autocast:
            processed = self.preprocessor(observation)
            batch = _repeat_batch(processed, num_candidates)
            state = batch.get("observation.state")
            if not torch.is_tensor(state):
                raise ValueError("preprocessed observation must contain tensor 'observation.state'")
            if noise is None:
                noise = self.make_noise(processed, num_candidates, generator=generator)
            self._validate_noise(noise, num_candidates, state)
            kwargs: dict[str, Any] = {}
            if guided:
                self._validate_guidance(
                    prefix_actions,
                    num_candidates=num_candidates,
                    state=state,
                    inference_delay=inference_delay,
                    execution_horizon=execution_horizon,
                )
                kwargs = {
                    "prev_chunk_left_over": prefix_actions,
                    "inference_delay": inference_delay,
                    "execution_horizon": execution_horizon,
                }
            elif inference_delay is not None or execution_horizon is not None:
                raise ValueError("RTC timing arguments require prefix_actions")
            self.last_noise = noise.detach().clone()
            policy_actions = self.policy.predict_action_chunk(batch, noise=noise, **kwargs)

        self._validate_policy_actions(policy_actions, num_candidates)
        with torch.inference_mode():
            control_actions = self.postprocessor(policy_actions.clone())
        if not torch.is_tensor(control_actions) or control_actions.ndim != 3:
            raise ValueError("policy postprocessor must return [N,H,A_control]")
        if control_actions.shape[:2] != policy_actions.shape[:2]:
            raise ValueError("policy postprocessor must preserve candidate count and horizon")
        if not control_actions.is_floating_point() or not bool(
            torch.isfinite(control_actions).all()
        ):
            raise ValueError("postprocessed control actions must be finite floating-point values")
        return policy_actions, control_actions.to(policy_actions.device)

    def _validate_noise(self, noise: Tensor, count: int, state: Tensor) -> None:
        expected = (
            count,
            int(self.policy.config.chunk_size),
            int(self.policy.config.max_action_dim),
        )
        if not torch.is_tensor(noise) or tuple(noise.shape) != expected:
            raise ValueError(f"noise must have shape {expected}")
        if noise.device != state.device or noise.dtype != torch.float32:
            raise ValueError("noise must be float32 on the preprocessed observation device")
        if not bool(torch.isfinite(noise).all()):
            raise ValueError("noise contains NaN or Inf")

    def _validate_policy_actions(self, actions: Tensor, count: int) -> None:
        expected = (count, int(self.policy.config.chunk_size))
        if not torch.is_tensor(actions) or actions.ndim != 3 or actions.shape[:2] != expected:
            raise ValueError(f"SmolVLA must return [N,H,A] with [N,H]={expected}")
        if actions.shape[-1] <= 0 or not actions.is_floating_point():
            raise ValueError("SmolVLA actions require a floating action dimension")
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("sampled policy actions contain NaN or Inf")

    def _validate_guidance(
        self,
        prefix: Tensor,
        *,
        num_candidates: int,
        state: Tensor,
        inference_delay: int | None,
        execution_horizon: int | None,
    ) -> None:
        if not torch.is_tensor(prefix) or prefix.ndim != 3:
            raise ValueError("prefix_actions must have shape [N,G,A_policy]")
        if prefix.shape[0] != num_candidates or prefix.shape[1] <= 0:
            raise ValueError("prefix_actions must contain one non-empty prefix per candidate")
        if prefix.shape[1] > int(self.policy.config.chunk_size):
            raise ValueError("prefix length cannot exceed the SmolVLA chunk size")
        if prefix.shape[2] <= 0 or prefix.shape[2] > int(self.policy.config.max_action_dim):
            raise ValueError("prefix policy-action dimension is invalid")
        if prefix.device != state.device or not prefix.is_floating_point():
            raise ValueError("prefix_actions must be floating point on the observation device")
        if not bool(torch.isfinite(prefix).all()):
            raise ValueError("prefix_actions contains NaN or Inf")
        if inference_delay is None or inference_delay < 0:
            raise ValueError("guided generation requires a non-negative inference_delay")
        if execution_horizon is None or execution_horizon <= 0:
            raise ValueError("guided generation requires a positive execution_horizon")
        if execution_horizon != prefix.shape[1]:
            raise ValueError("execution_horizon must equal the exact prefix length")
        rtc_config = getattr(self.policy.config, "rtc_config", None)
        if rtc_config is None or not bool(getattr(rtc_config, "enabled", False)):
            raise ValueError("RTC prefix guidance requires an enabled policy rtc_config")
        if getattr(self.policy, "rtc_processor", None) is None:
            raise ValueError("RTC prefix guidance requires an initialized RTC processor")
