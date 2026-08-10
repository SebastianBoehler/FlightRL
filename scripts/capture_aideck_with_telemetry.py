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
    load_fresh_flow_preflight_report,
)
from flightrl.hardware.paired_capture_contract import (
    CLEANUP_TIMEOUT_S,
    MINIMUM_CAMERA_RATE_HZ,
    MINIMUM_BATTERY_V,
    OVERALL_TIMEOUT_S,
    PROCESS_SCHEMA,
    REQUIRED_CAMERA_BIND_PORT,
    REQUIRED_CAMERA_FRAMES,
    REQUIRED_CAMERA_SOURCE_ENDPOINT,
    REQUIRED_DECK_EXPECTATIONS,
    REQUIRED_TELEMETRY,
    REQUIRED_TELEMETRY_COLUMNS,
    REQUIRED_TELEMETRY_URI,
    TELEMETRY_PERIOD_MS,
    TELEMETRY_DURATION_S,
    TELEMETRY_READY_TIMEOUT_S,
    TELEMETRY_TAIL_S,
)
from flightrl.hardware.paired_capture_process import run_bounded_capture_processes
from flightrl.hardware.telemetry import validate_log_duration


SCHEMA = PROCESS_SCHEMA
MOTION_START_SOUND = AUDIBLE_CUES["motion_start"]
SUCCESS_SOUND = AUDIBLE_CUES["success"]
FAILURE_SOUND = AUDIBLE_CUES["failure"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture AI Deck frames plus one-block USB telemetry without actuation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=TELEMETRY_DURATION_S)
    parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--host", default="192.168.4.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--bind-port", type=int, default=5001)
    parser.add_argument("--flow-preflight-report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        validate_log_duration(args.duration_s)
    except ValueError as exc:
        parser.error(str(exc))
    if args.frames != REQUIRED_CAMERA_FRAMES:
        parser.error(f"paired capture requires exactly {REQUIRED_CAMERA_FRAMES} frames")
    if args.duration_s != TELEMETRY_DURATION_S:
        parser.error(
            f"paired capture requires exactly {TELEMETRY_DURATION_S:g} seconds of telemetry"
        )

    config = load_hardware_config(args.config)
    if config.radio.uri != REQUIRED_TELEMETRY_URI:
        parser.error(f"paired capture requires the exact {REQUIRED_TELEMETRY_URI} URI")
    if (
        args.host != REQUIRED_CAMERA_SOURCE_ENDPOINT["host"]
        or args.port != REQUIRED_CAMERA_SOURCE_ENDPOINT["port"]
        or args.bind_port != REQUIRED_CAMERA_BIND_PORT
    ):
        parser.error(
            "paired capture requires the exact AI Deck UDP endpoint "
            f"{REQUIRED_CAMERA_SOURCE_ENDPOINT['host']}:"
            f"{REQUIRED_CAMERA_SOURCE_ENDPOINT['port']} and bind port "
            f"{REQUIRED_CAMERA_BIND_PORT}"
        )
    if any(
        getattr(config.decks, name) is not expected
        for name, expected in REQUIRED_DECK_EXPECTATIONS.items()
    ):
        parser.error(
            "paired capture requires the exact AI Deck, Flow Deck, and Z-ranger stack"
        )
    if (
        tuple(config.logging.variables) != REQUIRED_TELEMETRY
        or config.logging.period_ms != TELEMETRY_PERIOD_MS
    ):
        parser.error(
            "paired capture requires the exact five-variable stationary profile at 50 ms"
        )
    minimum_duration_s = (
        (args.frames - 1) / MINIMUM_CAMERA_RATE_HZ + TELEMETRY_TAIL_S
    )
    if args.duration_s < minimum_duration_s:
        parser.error(
            "telemetry duration must cover the minimum accepted camera rate plus tail "
            f"({minimum_duration_s:.3f} s required)"
        )

    preflight_evidence = None
    preflight_bytes = None
    afplay = None
    if not args.dry_run:
        if args.flow_preflight_report is None:
            parser.error("paired capture requires a fresh passing props-off Flow preflight")
        try:
            _preflight, preflight_evidence, preflight_bytes = (
                load_fresh_flow_preflight_report(
                    args.flow_preflight_report,
                    now_s=time(),
                )
            )
        except HardwareError as exc:
            parser.error(
                "paired capture requires a fresh passing props-off Flow preflight: "
                f"{exc}"
            )
        afplay = shutil.which("afplay")
        if afplay is None:
            parser.error("live paired capture requires afplay for audible cues")

    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    camera_output = run_dir / "decoded_frames.npz"
    telemetry_output = run_dir / "telemetry.csv"
    console_output = run_dir / "console.jsonl"
    camera_command = (
        sys.executable,
        str(root / "scripts" / "capture_aideck_vision.py"),
        "--transport",
        "udp",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--bind-port",
        str(args.bind_port),
        "--timeout-s",
        "5",
        "--frames",
        str(args.frames),
        "--output",
        str(camera_output),
    )
    telemetry_command = (
        sys.executable,
        str(root / "scripts" / "crazyflie_log.py"),
        "--config",
        str(args.config.resolve()),
        "--duration-s",
        str(args.duration_s),
        "--output",
        str(telemetry_output),
        "--console-output",
        str(console_output),
    )
    overall_timeout_s = OVERALL_TIMEOUT_S
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "controls_drone": False,
        "non_actuating": True,
        "telemetry_uri": config.radio.uri,
        "telemetry_period_ms": config.logging.period_ms,
        "telemetry_duration_s": args.duration_s,
        "telemetry_variables": list(config.logging.variables),
        "telemetry_log_blocks": 1,
        "camera_requested_frames": args.frames,
        "camera_source_endpoint": REQUIRED_CAMERA_SOURCE_ENDPOINT,
        "camera_bind_port": REQUIRED_CAMERA_BIND_PORT,
        "camera_command": list(camera_command),
        "telemetry_command": list(telemetry_command),
        "telemetry_ready_path": str(telemetry_output),
        "telemetry_ready_columns": list(REQUIRED_TELEMETRY_COLUMNS),
        "telemetry_ready_minimums": {"pm.vbat": MINIMUM_BATTERY_V},
        "telemetry_ready_timeout_s": TELEMETRY_READY_TIMEOUT_S,
        "cleanup_timeout_s": CLEANUP_TIMEOUT_S,
        "overall_timeout_s": overall_timeout_s,
        "flow_preflight_required_for_live": True,
        "audible_cues": AUDIBLE_CUES,
        "synchronization_authority": False,
        "authority_reason": (
            "Host epoch timestamps permit offline alignment, but this process manifest "
            "does not prove camera source sequence, checksums, or device-clock synchronization."
        ),
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    if preflight_evidence is None or preflight_bytes is None:
        raise RuntimeError("validated Flow preflight evidence is unavailable")
    if afplay is None:
        raise RuntimeError("validated audible cue player is unavailable")
    (run_dir / str(preflight_evidence["embedded_name"])).write_bytes(preflight_bytes)
    manifest["flow_preflight_evidence"] = preflight_evidence
    manifest["started_host_time_s"] = time()
    try:
        with (run_dir / "camera.log").open("w") as camera_log, (
            run_dir / "telemetry.log"
        ).open("w") as telemetry_log:
            outcome = run_bounded_capture_processes(
                camera_command=camera_command,
                telemetry_command=telemetry_command,
                telemetry_ready_path=telemetry_output,
                telemetry_required_columns=REQUIRED_TELEMETRY_COLUMNS,
                telemetry_minimum_values={"pm.vbat": MINIMUM_BATTERY_V},
                telemetry_ready_timeout_s=TELEMETRY_READY_TIMEOUT_S,
                timeout_s=overall_timeout_s,
                cleanup_timeout_s=CLEANUP_TIMEOUT_S,
                before_camera=lambda: _play_cue(afplay, MOTION_START_SOUND),
                camera_output=camera_log,
                telemetry_output=telemetry_log,
            )
    except Exception as exc:
        manifest["process_outcome"] = {
            "succeeded": False,
            "timed_out": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        succeeded = False
    else:
        manifest["process_outcome"] = {
            "succeeded": outcome.succeeded,
            "timed_out": outcome.timed_out,
            "elapsed_s": outcome.elapsed_s,
            "camera": {
                "pid": outcome.camera.pid,
                "returncode": outcome.camera.returncode,
            },
            "telemetry": {
                "pid": outcome.telemetry.pid,
                "returncode": outcome.telemetry.returncode,
            },
        }
        succeeded = outcome.succeeded
    try:
        _play_cue(afplay, SUCCESS_SOUND if succeeded else FAILURE_SOUND)
    except Exception as exc:
        manifest["audible_end_cue_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        manifest["process_outcome"]["succeeded"] = False
        succeeded = False
    manifest["ended_host_time_s"] = time()
    _write_manifest(run_dir, manifest)
    if not succeeded:
        print(f"paired capture failed; inspect {run_dir / 'capture_process.json'}", file=sys.stderr)
        return 2
    print(f"paired capture completed: {run_dir}")
    return 0


def _write_manifest(run_dir: Path, manifest: dict[str, object]) -> None:
    (run_dir / "capture_process.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _play_cue(afplay: str, sound: str) -> None:
    subprocess.run(
        (afplay, sound),
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5.0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
