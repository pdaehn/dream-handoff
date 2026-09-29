"""Narrow compatibility bridge for DreamHandoff on the pinned LeRobot rollout API.

LeRobot commit ``c841a0c25833b866970c5a31d66c2bff60f171de`` only constructs its
built-in inference engines.  We therefore let it construct an unstarted stock
sync engine, replace that construction-only placeholder, and restore the real
configuration before a rollout strategy can observe or start inference.

The sync engine is never a runtime wrapper around DreamHandoff.  This module can
be deleted once LeRobot supports direct third-party inference construction.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from copy import copy
from threading import Event
from typing import Any

from lerobot.rollout import (
    RolloutConfig,
    RolloutContext,
    SyncInferenceConfig,
    SyncInferenceEngine,
    build_rollout_context,
)

from dream_handoff.inference import (
    DreamHandoffInferenceConfig,
    DreamHandoffInferenceEngine,
)
from dream_handoff.r2dreamer import R2DreamerRuntime

logger = logging.getLogger(__name__)


def validate_dreamhandoff_rollout_config(cfg: RolloutConfig) -> None:
    """Reject policy compilation until this compatibility lifecycle supports it."""

    if bool(getattr(cfg, "use_torch_compile", False)):
        raise ValueError(
            "Policy torch compilation is unsupported by the DreamHandoff rollout "
            "compatibility path; set use_torch_compile=false"
        )


def _feature_dimension(feature: Any) -> int | None:
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, Mapping):
        shape = feature.get("shape")
    if not shape or len(shape) != 1:
        return None
    return int(shape[0])


def _validate_r2_before_hardware(
    runtime: R2DreamerRuntime,
    cfg: RolloutConfig,
) -> None:
    """Check checkpoint/policy facts available before LeRobot connects hardware."""
    policy_cfg = cfg.policy
    if policy_cfg is None:
        raise ValueError("DreamHandoff rollout requires a loaded policy configuration")

    policy_action_names = getattr(policy_cfg, "action_feature_names", None)
    if policy_action_names and tuple(policy_action_names) != tuple(runtime.action_names):
        raise ValueError("R2 canonical action names do not match the policy action order")

    output_features = getattr(policy_cfg, "output_features", {}) or {}
    action_dimension = _feature_dimension(output_features.get("action"))
    if action_dimension is not None and action_dimension != len(runtime.action_names):
        raise ValueError("R2 canonical action dimension does not match the policy action dimension")

    input_features = getattr(policy_cfg, "input_features", {}) or {}
    available_observations = set(input_features)
    # A rename map's source keys are the dataset-frame keys seen directly by R2;
    # its destination keys are the policy-facing names.
    available_observations.update(cfg.rename_map)
    missing = sorted(set(runtime.observation_keys) - available_observations)
    if input_features and missing:
        raise ValueError(f"R2 checkpoint requires unavailable policy observations {missing}")


def _validate_sync_placeholder(ctx: RolloutContext) -> SyncInferenceEngine:
    """Require the exact resource-free stock placeholder built by pinned LeRobot.

    This LeRobot revision exposes no started flag.  Its ``SyncInferenceEngine``
    owns no thread or worker, and ``start()`` is a no-op; pinned strategy source
    first calls it from ``strategy.setup``.  Exact type and ownership checks here,
    plus call-order regression tests, establish that replacement happens before
    any lifecycle call.
    """
    placeholder = ctx.policy.inference
    if type(placeholder) is not SyncInferenceEngine:
        raise TypeError(
            "LeRobot context did not contain the expected stock SyncInferenceEngine placeholder"
        )
    expected_objects = {
        "_policy": ctx.policy.policy,
        "_preprocessor": ctx.policy.preprocessor,
        "_postprocessor": ctx.policy.postprocessor,
        "_dataset_features": ctx.data.dataset_features,
        "_ordered_action_keys": ctx.data.ordered_action_keys,
    }
    for attribute, expected in expected_objects.items():
        if getattr(placeholder, attribute, None) is not expected:
            raise ValueError(f"stock sync placeholder has unexpected {attribute} ownership")
    return placeholder


def _disconnect_failed_context(ctx: RolloutContext) -> None:
    """Release connected hardware without dispatching or returning the robot."""
    devices = (
        ("robot", ctx.hardware.robot_wrapper.inner),
        ("teleoperator", ctx.hardware.teleop),
    )
    errors: list[tuple[str, BaseException]] = []
    for name, device in devices:
        if device is None:
            continue
        try:
            if device.is_connected:
                device.disconnect()
        except BaseException as exc:
            errors.append((name, exc))
    if errors:
        primary_name, primary = errors[0]
        primary.add_note(f"DreamHandoff could not disconnect {primary_name}")
        for name, error in errors[1:]:
            primary.add_note(f"DreamHandoff could not disconnect {name}: {error!r}")
        raise primary


def disconnect_rollout_context_without_actions(ctx: RolloutContext) -> None:
    """Release a built context when setup fails before strategy ownership begins."""

    _disconnect_failed_context(ctx)


def build_dreamhandoff_rollout_context(
    cfg: RolloutConfig,
    shutdown_event: Event,
    *,
    event_sink: Any | None = None,
) -> RolloutContext:
    """Build one LeRobot context and install the existing DreamHandoff engine.

    R2 is loaded before ``build_rollout_context`` because that upstream call
    connects the robot.  ``event_sink`` is the existing WP3 observation seam
    reserved for WP5 capture; this module does not implement capture.
    """
    inference_config = cfg.inference
    if not isinstance(inference_config, DreamHandoffInferenceConfig):
        raise TypeError("rollout requires --inference.type=dream_handoff")
    validate_dreamhandoff_rollout_config(cfg)

    logger.info("Loading and validating R2 before LeRobot hardware construction...")
    r2_runtime = R2DreamerRuntime.from_checkpoint(
        inference_config.r2_checkpoint,
        device=cfg.device or "cpu",
    )
    _validate_r2_before_hardware(r2_runtime, cfg)

    placeholder_cfg = copy(cfg)
    placeholder_cfg.inference = SyncInferenceConfig()
    logger.info("Building one LeRobot context with a construction-only sync placeholder...")
    ctx = build_rollout_context(placeholder_cfg, shutdown_event)

    try:
        if type(ctx.runtime.cfg.inference) is not SyncInferenceConfig:
            raise ValueError("LeRobot context did not retain the temporary sync configuration")
        _validate_sync_placeholder(ctx)

        task = (
            ctx.runtime.cfg.dataset.single_task if ctx.runtime.cfg.dataset else ctx.runtime.cfg.task
        )
        engine = DreamHandoffInferenceEngine(
            config=inference_config,
            policy=ctx.policy.policy,
            preprocessor=ctx.policy.preprocessor,
            postprocessor=ctx.policy.postprocessor,
            r2_runtime=r2_runtime,
            dataset_features=ctx.data.dataset_features,
            ordered_action_keys=ctx.data.ordered_action_keys,
            task=task,
            fps=float(ctx.runtime.cfg.fps),
            device=ctx.runtime.cfg.device,
            robot_type=ctx.hardware.robot_wrapper.robot_type,
            shutdown_event=ctx.runtime.shutdown_event,
            event_sink=event_sink,
        )
        ctx.policy.inference = engine
        ctx.runtime.cfg.inference = inference_config
    except BaseException as primary:
        try:
            _disconnect_failed_context(ctx)
        except BaseException as cleanup_error:
            primary.add_note(f"DreamHandoff context cleanup also failed: {cleanup_error!r}")
            logger.exception("Could not disconnect hardware after inference injection failed")
        raise
    logger.info("Installed DreamHandoff inference and restored its truthful configuration")
    return ctx


__all__ = [
    "build_dreamhandoff_rollout_context",
    "disconnect_rollout_context_without_actions",
    "validate_dreamhandoff_rollout_config",
]
