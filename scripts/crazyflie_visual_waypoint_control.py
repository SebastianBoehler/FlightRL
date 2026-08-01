from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

from flightrl.hardware.visual_waypoint import (
    VisualWaypointConfig,
    require_visual_live_readiness,
)
from flightrl.hardware.visual_waypoint_flight import (
    VisualFlightConfig,
    run_visual_waypoint_flight,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "puffer_visual"
DEFAULT_CHECKPOINT = ARTIFACTS / "flightrl_visual_fast16_lowlight_v5_1048576.bin"
DEFAULT_TRAINING_REPORT = DEFAULT_CHECKPOINT.with_suffix(".report.json")
DEFAULT_SHADOW_REPORT = (
    ARTIFACTS
    / "flightrl_visual_fast16_lowlight_v5_policy_profile_gate_final.summary.json"
)


def main() -> None:
    args = parse_args()
    waypoint = VisualWaypointConfig(
        distance_m=args.distance_m,
        height_m=args.height_m,
        base_speed_m_s=args.base_speed_m_s,
        policy_blend=args.policy_blend,
        max_lateral_residual_m_s=args.max_lateral_residual_m_s,
        max_total_speed_m_s=args.max_total_speed_m_s,
        max_displacement_m=args.max_displacement_m,
    )
    live = VisualFlightConfig(
        baseline_hover_only=args.baseline_hover_only,
        waypoint_count=args.waypoint_count,
        settle_hover_s=args.settle_hover_s,
        max_hover_displacement_m=args.max_hover_displacement_m,
        max_hover_speed_m_s=args.max_hover_speed_m_s,
        max_active_s=args.max_active_s,
        warmup_frames=args.warmup_frames,
        warmup_timeout_s=args.warmup_timeout_s,
        camera_timeout_s=args.camera_timeout_s,
        max_camera_age_s=args.max_camera_age_s,
        max_dropped_frames=args.max_dropped_frames,
        min_frame_mean=args.min_frame_mean,
        min_input_contrast=args.min_input_contrast,
        min_battery_v=args.min_battery_v,
        log_timeout_s=args.log_timeout_s,
    )
    readiness = require_visual_live_readiness(
        args.checkpoint,
        args.training_report,
        args.shadow_report,
    )
    if args.dry_run:
        print_plan(readiness, waypoint, live)
        return
    require_confirmations(args)
    summary, rows = run_visual_waypoint_flight(
        args.checkpoint,
        args.hardware_config,
        waypoint,
        live,
        readiness,
    )
    write_rows(args.output, rows)
    summary_path = Path(args.output).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"output={Path(args.output).resolve()}")
    print(f"summary={summary_path.resolve()}")


def print_plan(readiness, waypoint, live) -> None:
    print(
        json.dumps(
            {
                **readiness,
                "controls_drone": False,
                "policy_axis_authority": "lateral_only",
                "waypoint": asdict(waypoint),
                "live_gates": asdict(live),
            },
            indent=2,
            sort_keys=True,
        )
    )


def write_rows(path, rows) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["host_time_s"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require_confirmations(args) -> None:
    required = (
        args.confirm_flight,
        args.confirm_clear_path,
        args.confirm_room_lit,
    )
    if not all(required) or (
        not args.baseline_hover_only
        and not args.confirm_visual_policy_control
    ):
        raise SystemExit(
            "live flight requires --confirm-flight, --confirm-clear-path, "
            "and --confirm-room-lit; active policy control additionally "
            "requires --confirm-visual-policy-control"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded lateral visual-policy authority toward one straight waypoint"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--training-report", type=Path, default=DEFAULT_TRAINING_REPORT)
    parser.add_argument("--shadow-report", type=Path, default=DEFAULT_SHADOW_REPORT)
    parser.add_argument(
        "--hardware-config",
        default="configs/hardware/crazyflie_2_1_brushless_flow_only.toml",
    )
    parser.add_argument(
        "--output",
        default="artifacts/crazyflie_logs/visual_waypoint_control.csv",
    )
    parser.add_argument("--distance-m", type=float, default=0.30)
    parser.add_argument("--height-m", type=float, default=0.55)
    parser.add_argument("--base-speed-m-s", type=float, default=0.06)
    parser.add_argument("--policy-blend", type=float, default=0.40)
    parser.add_argument("--max-lateral-residual-m-s", type=float, default=0.02)
    parser.add_argument("--max-total-speed-m-s", type=float, default=0.09)
    parser.add_argument("--max-displacement-m", type=float, default=0.45)
    parser.add_argument("--baseline-hover-only", action="store_true")
    parser.add_argument("--waypoint-count", type=int, default=1)
    parser.add_argument("--settle-hover-s", type=float, default=2.0)
    parser.add_argument("--max-hover-displacement-m", type=float, default=0.08)
    parser.add_argument("--max-hover-speed-m-s", type=float, default=0.12)
    parser.add_argument("--max-active-s", type=float, default=8.0)
    parser.add_argument("--warmup-frames", type=int, default=64)
    parser.add_argument("--warmup-timeout-s", type=float, default=5.0)
    parser.add_argument("--camera-timeout-s", type=float, default=3.0)
    parser.add_argument("--max-camera-age-s", type=float, default=0.25)
    parser.add_argument("--max-dropped-frames", type=int, default=5)
    parser.add_argument("--min-frame-mean", type=float, default=8.0)
    parser.add_argument("--min-input-contrast", type=float, default=0.10)
    parser.add_argument("--min-battery-v", type=float, default=3.55)
    parser.add_argument("--log-timeout-s", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-flight", action="store_true")
    parser.add_argument("--confirm-visual-policy-control", action="store_true")
    parser.add_argument("--confirm-clear-path", action="store_true")
    parser.add_argument("--confirm-room-lit", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
