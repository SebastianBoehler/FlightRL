from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import time

from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.errors import HardwareError
from flightrl.hardware.flow_preflight_contract import (
    AUDIBLE_CUES,
    CLEANUP_TIMEOUT_S,
    DECK_CHECK_TIMEOUT_S,
    PROCESS_SCHEMA,
    REQUIRED_DECK_EXPECTATIONS,
    REQUIRED_TELEMETRY,
    TELEMETRY_DURATION_S,
    TELEMETRY_TIMEOUT_S,
    validate_flow_preflight_config,
)
from flightrl.hardware.flow_preflight_process import run_flow_preflight_processes


SCHEMA = PROCESS_SCHEMA
MOTION_START_SOUND = AUDIBLE_CUES["motion_start"]
SUCCESS_SOUND = AUDIBLE_CUES["success"]
FAILURE_SOUND = AUDIBLE_CUES["failure"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a bounded, props-off, non-actuating AI Deck/Flow preflight."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_hardware_config(args.config)
        validate_flow_preflight_config(config)
    except (OSError, HardwareError) as exc:
        parser.error(str(exc))

    root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    telemetry_path = run_dir / "flow_preflight.csv"
    check_command = (
        sys.executable,
        str(root / "scripts" / "check_aideck_flow_preflight.py"),
        "--config",
        str(config_path),
    )
    telemetry_command = (
        sys.executable,
        str(root / "scripts" / "crazyflie_log.py"),
        "--config",
        str(config_path),
        "--duration-s",
        str(TELEMETRY_DURATION_S),
        "--output",
        str(telemetry_path),
        "--console-output",
        str(run_dir / "console.jsonl"),
    )
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "controls_drone": False,
        "non_actuating": True,
        "props_off_required": True,
        "rigid_support_required": True,
        "telemetry_uri": config.radio.uri,
        "deck_expectations": REQUIRED_DECK_EXPECTATIONS,
        "telemetry_variables": list(REQUIRED_TELEMETRY),
        "telemetry_period_ms": config.logging.period_ms,
        "telemetry_log_blocks": 1,
        "deck_check_command": list(check_command),
        "telemetry_command": list(telemetry_command),
        "deck_check_timeout_s": DECK_CHECK_TIMEOUT_S,
        "telemetry_duration_s": TELEMETRY_DURATION_S,
        "telemetry_timeout_s": TELEMETRY_TIMEOUT_S,
        "cleanup_timeout_s": CLEANUP_TIMEOUT_S,
        "audible_cues": {
            "motion_start": MOTION_START_SOUND,
            "success": SUCCESS_SOUND,
            "failure": FAILURE_SOUND,
        },
        "flow_validation_path": str(run_dir / "flow_validation.json"),
        "flight_authority": False,
        "authority_reason": (
            "This path only checks a props-off deck/TOC contract and raw Flow/Z-ranger "
            "response; it cannot grant estimator, policy, deployment, or flight authority."
        ),
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    afplay = shutil.which("afplay")
    if afplay is None:
        parser.error("live Flow preflight requires afplay for audible motion cues")

    run_dir.mkdir(parents=True, exist_ok=False)
    manifest["started_host_time_s"] = time()
    print("PROPS-OFF DECK CHECK START", flush=True)
    try:
        with (run_dir / "deck_check.log").open("w") as deck_log, (
            run_dir / "telemetry.log"
        ).open("w") as telemetry_log:
            outcome = run_flow_preflight_processes(
                deck_check_command=check_command,
                telemetry_command=telemetry_command,
                telemetry_path=telemetry_path,
                telemetry_log_path=run_dir / "telemetry.log",
                deck_check_timeout_s=DECK_CHECK_TIMEOUT_S,
                telemetry_timeout_s=TELEMETRY_TIMEOUT_S,
                cleanup_timeout_s=CLEANUP_TIMEOUT_S,
                before_telemetry=lambda: _play_cue(afplay, MOTION_START_SOUND),
                deck_check_output=deck_log,
                telemetry_output=telemetry_log,
            )
        manifest["process_outcome"] = _outcome_dict(outcome)
        if outcome.validation is not None:
            (run_dir / "flow_validation.json").write_text(
                json.dumps(outcome.validation, indent=2, sort_keys=True) + "\n"
            )
        succeeded = outcome.succeeded
    except Exception as exc:
        manifest["process_outcome"] = {
            "succeeded": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        succeeded = False
    try:
        _play_cue(afplay, SUCCESS_SOUND if succeeded else FAILURE_SOUND)
    except Exception as exc:
        manifest["audible_end_cue_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        succeeded = False
        manifest["process_outcome"]["succeeded"] = False
    manifest["ended_host_time_s"] = time()
    (run_dir / "preflight_process.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if not succeeded:
        print(f"preflight failed; inspect {run_dir / 'preflight_process.json'}", file=sys.stderr)
        return 2
    print(f"props-off preflight completed: {run_dir}")
    return 0


def _play_cue(afplay: str, sound: str) -> None:
    subprocess.run(
        (afplay, sound),
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5.0,
    )


def _outcome_dict(outcome) -> dict[str, object]:
    telemetry = outcome.telemetry
    return {
        "succeeded": outcome.succeeded,
        "deck_check": {
            "pid": outcome.deck_check.pid,
            "returncode": outcome.deck_check.returncode,
            "timed_out": outcome.deck_check.timed_out,
            "elapsed_s": outcome.deck_check.elapsed_s,
        },
        "telemetry": None
        if telemetry is None
        else {
            "pid": telemetry.pid,
            "returncode": telemetry.returncode,
            "timed_out": telemetry.timed_out,
            "elapsed_s": telemetry.elapsed_s,
        },
        "validation_error": outcome.validation_error,
        "packet_loss_free": outcome.packet_loss_free,
        "flow_preflight_passed": (
            None
            if outcome.validation is None
            else outcome.validation["flow_preflight_passed"]
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
