from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, IO, Sequence

from .flow_preflight_validation import validate_flow_preflight
from .paired_capture_process import BoundedProcessOutcome, run_bounded_process


@dataclass(frozen=True, slots=True)
class FlowPreflightProcessOutcome:
    deck_check: BoundedProcessOutcome
    telemetry: BoundedProcessOutcome | None
    validation: dict[str, object] | None
    packet_loss_free: bool | None
    validation_error: dict[str, str] | None = None

    @property
    def succeeded(self) -> bool:
        return (
            self.deck_check.succeeded
            and self.telemetry is not None
            and self.telemetry.succeeded
            and self.validation is not None
            and self.validation["flow_preflight_passed"] is True
            and self.packet_loss_free is True
            and self.validation_error is None
        )


def run_flow_preflight_processes(
    *,
    deck_check_command: Sequence[str],
    telemetry_command: Sequence[str],
    telemetry_path: str | Path,
    telemetry_log_path: str | Path,
    deck_check_timeout_s: float,
    telemetry_timeout_s: float,
    cleanup_timeout_s: float,
    before_telemetry: Callable[[], None],
    deck_check_output: IO[str] | int | None = None,
    telemetry_output: IO[str] | int | None = None,
) -> FlowPreflightProcessOutcome:
    deck_check = run_bounded_process(
        command=deck_check_command,
        timeout_s=deck_check_timeout_s,
        cleanup_timeout_s=cleanup_timeout_s,
        output=deck_check_output,
    )
    if not deck_check.succeeded:
        return FlowPreflightProcessOutcome(deck_check, None, None, None)

    before_telemetry()
    telemetry = run_bounded_process(
        command=telemetry_command,
        timeout_s=telemetry_timeout_s,
        cleanup_timeout_s=cleanup_timeout_s,
        output=telemetry_output,
    )
    packet_loss_free = _packet_loss_free(Path(telemetry_log_path))
    if not telemetry.succeeded:
        return FlowPreflightProcessOutcome(
            deck_check, telemetry, None, packet_loss_free
        )
    if not packet_loss_free:
        return FlowPreflightProcessOutcome(deck_check, telemetry, None, False)
    try:
        validation = validate_flow_preflight(telemetry_path)
    except Exception as exc:
        return FlowPreflightProcessOutcome(
            deck_check,
            telemetry,
            None,
            packet_loss_free,
            {"type": type(exc).__name__, "message": str(exc)},
        )
    return FlowPreflightProcessOutcome(
        deck_check, telemetry, validation, packet_loss_free
    )


def _packet_loss_free(path: Path) -> bool:
    try:
        text = path.read_text(errors="replace").lower()
    except OSError:
        return False
    return "too many packets lost" not in text
