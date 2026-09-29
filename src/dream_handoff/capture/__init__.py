"""Optional format-16 research capture outside controller semantics."""

from .rollout import (
    DiagnosticActionProcessor,
    DiagnosticRobotWrapper,
    RolloutCapture,
    install_diagnostic_capture,
)
from .schema import (
    DIAGNOSTIC_FORMAT_VERSION,
    DREAMHANDOFF_DIAGNOSTIC_FORMAT,
    HISTORICAL_DIAGNOSTIC_FORMAT,
    DiagnosticCapture,
    HandoffRecord,
    RequestRecord,
    load_diagnostic_capture,
    sha256_file,
)

__all__ = [
    "DIAGNOSTIC_FORMAT_VERSION",
    "DREAMHANDOFF_DIAGNOSTIC_FORMAT",
    "DiagnosticActionProcessor",
    "DiagnosticCapture",
    "DiagnosticRobotWrapper",
    "HISTORICAL_DIAGNOSTIC_FORMAT",
    "HandoffRecord",
    "RequestRecord",
    "RolloutCapture",
    "install_diagnostic_capture",
    "load_diagnostic_capture",
    "sha256_file",
]
