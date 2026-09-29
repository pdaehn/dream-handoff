#!/usr/bin/env python
"""Run DreamHandoff through the stock pinned-LeRobot rollout strategy."""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from lerobot.configs import parser
from lerobot.rollout import RolloutConfig, RolloutContext, create_strategy
from lerobot.scripts import lerobot_rollout as _lerobot_rollout  # noqa: F401
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization

from dream_handoff.capture import RolloutCapture, install_diagnostic_capture
from dream_handoff.inference import DreamHandoffInferenceConfig
from dream_handoff.lerobot_compat import (
    build_dreamhandoff_rollout_context,
    disconnect_rollout_context_without_actions,
    validate_dreamhandoff_rollout_config,
)

logger = logging.getLogger(__name__)


@dataclass
class DreamHandoffRolloutConfig(RolloutConfig):
    """Stock rollout configuration with one explicit optional capture destination."""

    capture: Path | None = None


def run_rollout(cfg: RolloutConfig, shutdown_event: Event) -> RolloutContext:
    """Build the compatibility context and run the stock strategy lifecycle."""
    validate_dreamhandoff_rollout_config(cfg)
    capture_path = getattr(cfg, "capture", None)
    capture = None if capture_path is None else RolloutCapture(capture_path)
    capture_installed = False
    ctx = None
    strategy = None
    try:
        if capture is None:
            ctx = build_dreamhandoff_rollout_context(cfg, shutdown_event)
        else:
            ctx = build_dreamhandoff_rollout_context(
                cfg,
                shutdown_event,
                event_sink=capture.event_sink,
            )
            install_diagnostic_capture(ctx, capture)
            capture_installed = True
            logger.info("Format-16 diagnostics will be saved to %s", capture.output_path)

        strategy = create_strategy(cfg.strategy)
        rollout_error = None
        try:
            strategy.setup(ctx)
            logger.info("Rollout setup complete; starting the stock LeRobot strategy")
            strategy.run(ctx)
        except BaseException as exc:
            rollout_error = exc
            if capture is not None:
                capture.record_rollout_failure(exc)
            raise
        finally:
            try:
                strategy.teardown(ctx)
            except BaseException as exc:
                if capture is not None:
                    capture.record_rollout_failure(exc)
                if rollout_error is None:
                    raise
                logger.exception("Rollout teardown also failed; preserving the rollout error")
        return ctx
    except BaseException as exc:
        if capture is not None:
            capture.record_rollout_failure(exc)
        if ctx is not None and strategy is None:
            try:
                disconnect_rollout_context_without_actions(ctx)
            except BaseException as cleanup_error:
                exc.add_note(f"DreamHandoff context cleanup also failed: {cleanup_error!r}")
                logger.exception("Could not disconnect after capture/strategy setup failed")
        raise
    finally:
        if capture is not None and capture_installed:
            active_error = sys.exception()
            try:
                saved = capture.save()
                logger.info("Saved format-16 diagnostic capture to %s", saved)
            except Exception:
                if active_error is None:
                    raise
                logger.exception("Could not save diagnostic capture; preserving rollout failure")


@parser.wrap()
def rollout(cfg: DreamHandoffRolloutConfig) -> None:
    """Parse LeRobot rollout configuration and install DreamHandoff inference."""
    init_logging()
    if not isinstance(cfg.inference, DreamHandoffInferenceConfig):
        raise ValueError("DreamHandoff rollout requires --inference.type=dream_handoff")
    logger.info("Resolved DreamHandoff rollout configuration:\n%s", cfg)
    logger.info("Resolved diagnostic capture destination: %s", cfg.capture)

    if cfg.display_data:
        init_visualization(
            cfg.display_mode,
            session_name="dream-handoff-rollout",
            ip=cfg.display_ip,
            port=cfg.display_port,
        )

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    try:
        run_rollout(cfg, signal_handler.shutdown_event)
    except KeyboardInterrupt:
        logger.info("Interrupted by operator")
    finally:
        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)


def main() -> None:
    """CLI entry point."""
    register_third_party_plugins()
    rollout()


if __name__ == "__main__":
    main()
