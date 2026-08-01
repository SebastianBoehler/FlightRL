from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from flightrl.puffer4_door_live_evidence import (
    validate_fixed_door_live_evidence,
)
from flightrl.puffer4_door_runtime import DoorPufferShadow
from flightrl.puffer4_door_shadow_capture import (
    collect_live_rows,
    dry_run_row,
    latest_detection as latest_detection,
)
from flightrl.puffer4_door_shadow_detector_contract import (
    APPROVED_SHADOW_DETECTOR_MODEL_ID,
    APPROVED_SHADOW_DEVICE,
    APPROVED_SHADOW_HARDWARE_CONFIG,
    APPROVED_SHADOW_PROMPT,
    APPROVED_SHADOW_THRESHOLD,
    approved_shadow_hardware_config_snapshot,
)
from flightrl.puffer4_door_shadow_identity import (
    build_fixed_door_shadow_identity,
)
from flightrl.puffer4_door_shadow_io import (
    summarize_shadow_rows,
)
from flightrl.puffer4_door_shadow_projection import (
    bind_fixed_door_shadow_rows,
)
from flightrl.puffer4_door_snapshot import (
    load_fixed_door_checkpoint_snapshot,
)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT
    / "artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m"
    / "flightrl_fixed_door_d1_seed11_1048576.bin"
)


def main() -> None:
    args = parse_args()
    with exclusive_shadow_outputs(args.output) as summary_path:
        run_shadow(args, summary_path)


def run_shadow(args: argparse.Namespace, summary_path: Path) -> None:
    report_path = args.training_report or preferred_report(args.checkpoint)
    evidence = validate_fixed_door_live_evidence(
        args.checkpoint,
        report_path,
    )
    bundle = evidence.bundle
    identity = build_fixed_door_shadow_identity(
        evidence,
        prompt=args.prompt,
        detector_model_id=args.model_id,
        threshold=args.threshold,
        device=args.device,
        hardware_config=args.hardware_config,
    )
    snapshot = load_fixed_door_checkpoint_snapshot(
        args.checkpoint,
        bundle.checkpoint_sha256,
    )
    refreshed_evidence = validate_fixed_door_live_evidence(
        args.checkpoint,
        report_path,
    )
    refreshed_identity = build_fixed_door_shadow_identity(
        refreshed_evidence,
        prompt=args.prompt,
        detector_model_id=args.model_id,
        threshold=args.threshold,
        device=args.device,
        hardware_config=args.hardware_config,
    )
    if refreshed_identity != identity:
        raise ValueError(
            "fixed-door shadow evidence changed during policy snapshot"
        )
    shadow = DoorPufferShadow.from_state_dict(
        snapshot.state_dict,
        architecture=bundle.architecture,
    )
    if args.dry_run:
        rows = [dry_run_row(shadow)]
        dropped_frames = 0
    else:
        with approved_shadow_hardware_config_snapshot(
            args.hardware_config
        ) as hardware_snapshot:
            rows, dropped_frames = collect_live_rows(
                shadow,
                args,
                hardware_config_path=hardware_snapshot,
            )
    rows = bind_fixed_door_shadow_rows(
        rows,
        identity,
        bundle.action_contract,
    )
    write_rows(args.output, rows)
    summary = summarize_shadow_rows(
        rows,
        checkpoint=args.checkpoint,
        training_report=report_path,
        simulation_gate=bundle.raw_report["simulation_gate"],
        dropped_frames=dropped_frames,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"output={args.output}")
    print(f"summary={summary_path}")


def preferred_report(checkpoint: Path) -> Path:
    reevaluation = checkpoint.with_suffix(".reevaluation.json")
    return (
        reevaluation
        if reevaluation.exists()
        else checkpoint.with_suffix(".report.json")
    )


@contextmanager
def exclusive_shadow_outputs(path: Path) -> Iterator[Path]:
    summary_path = path.with_suffix(".summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        for candidate in (path, summary_path):
            try:
                candidate.open("x").close()
            except FileExistsError as exc:
                raise FileExistsError(
                    f"shadow output already exists: {candidate}"
                ) from exc
            created.append(candidate)
        yield summary_path
    except BaseException:
        for candidate in reversed(created):
            candidate.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description="Run the recurrent fixed-door checkpoint without issuing commands"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--prompt", default=APPROVED_SHADOW_PROMPT)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / f"artifacts/crazyflie_logs/door_puffer_shadow_{timestamp}.csv",
    )
    parser.add_argument(
        "--hardware-config",
        type=Path,
        default=APPROVED_SHADOW_HARDWARE_CONFIG,
    )
    parser.add_argument(
        "--model-id",
        default=APPROVED_SHADOW_DETECTOR_MODEL_ID,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps"),
        default=APPROVED_SHADOW_DEVICE,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=APPROVED_SHADOW_THRESHOLD,
    )
    parser.add_argument("--aideck-host", default="192.168.4.1")
    parser.add_argument("--aideck-port", type=int, default=5000)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--bind-port", type=int, default=5001)
    parser.add_argument("--camera-timeout-s", type=float, default=10.0)
    parser.add_argument("--log-timeout-s", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
