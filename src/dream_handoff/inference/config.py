"""Validated configuration for the one DreamHandoff inference engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lerobot.rollout import InferenceEngineConfig

AsyncBankGuidance = Literal["plain", "static_rtc"]
SelectorMode = Literal["baseline", "absolute_hysteresis"]

FINAL_HYSTERESIS_TAU = 0.06851652264595032


@InferenceEngineConfig.register_subclass("dream_handoff")
@dataclass
class DreamHandoffInferenceConfig(InferenceEngineConfig):
    """Final live inference behavior.

    Rollout concerns such as task, device, FPS, policy identity, cameras, and
    robot construction are deliberately constructor inputs to the engine.
    """

    r2_checkpoint: Path
    num_candidates: int = 10
    max_bank_age: int = 35
    bank_refill_threshold: int = 15
    async_bank_guidance: AsyncBankGuidance = "plain"
    guidance_horizon: int = 15
    compile_world_model_imagination: bool = True
    selector_mode: SelectorMode = "absolute_hysteresis"
    hysteresis_tau: float | None = FINAL_HYSTERESIS_TAU
    phase0_seed: int = 0

    def __post_init__(self) -> None:
        self.r2_checkpoint = Path(self.r2_checkpoint)
        if not str(self.r2_checkpoint):
            raise ValueError("r2_checkpoint must identify an external checkpoint")
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if self.max_bank_age <= 0:
            raise ValueError("max_bank_age must be positive")
        if self.bank_refill_threshold <= 0:
            raise ValueError("bank_refill_threshold must be positive")
        if self.bank_refill_threshold >= self.max_bank_age:
            raise ValueError("bank_refill_threshold must be smaller than max_bank_age")
        if self.async_bank_guidance not in {"plain", "static_rtc"}:
            raise ValueError("async_bank_guidance must be 'plain' or 'static_rtc'")
        if self.guidance_horizon <= 0:
            raise ValueError("guidance_horizon must be positive")
        if self.guidance_horizon > self.bank_refill_threshold:
            raise ValueError("guidance_horizon cannot exceed bank_refill_threshold")
        if self.selector_mode not in {"baseline", "absolute_hysteresis"}:
            raise ValueError("selector_mode must be 'baseline' or 'absolute_hysteresis'")
        if self.hysteresis_tau is not None and (
            self.hysteresis_tau < 0 or not math.isfinite(self.hysteresis_tau)
        ):
            raise ValueError("hysteresis_tau must be a finite non-negative distance")
        if self.selector_mode == "absolute_hysteresis" and self.hysteresis_tau is None:
            raise ValueError("absolute_hysteresis requires hysteresis_tau")
