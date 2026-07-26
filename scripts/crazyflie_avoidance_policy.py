from __future__ import annotations

import argparse

from flightrl.hardware.avoidance_policy import (
    AvoidanceCommand,
    RangerReading,
    reading_from_telemetry,
)
from flightrl.hardware.avoidance_live import (
    build_control_command,
    build_shadow_command,
    build_target_shadow_command,
    build_ttc_shadow_command,
    maybe_emergency_command,
    smooth_avoidance_command,
)
from flightrl.hardware.avoidance_runner import load_policy, require_policy_approval, run_live, write_rows
from flightrl.hardware.target_conditioned_policy import load_target_policy
from flightrl.hardware.ttc_policy import load_ttc_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained ranger avoidance policy on Crazyflie hover setpoints")
    parser.add_argument("--checkpoint")
    parser.add_argument("--shadow-checkpoint")
    parser.add_argument("--shadow-max-speed-m-s", type=float, default=None)
    parser.add_argument("--target-shadow-checkpoint")
    parser.add_argument("--target-shadow-max-speed-m-s", type=float, default=0.90)
    parser.add_argument("--ttc-shadow-checkpoint")
    parser.add_argument("--ttc-shadow-max-speed-m-s", type=float, default=0.65)
    parser.add_argument("--controller", choices=("policy", "ttc-policy", "reactive", "directional"), default="policy")
    parser.add_argument("--hardware-config", default="configs/hardware/crazyflie_2_1_brushless.toml")
    parser.add_argument("--output", default="artifacts/crazyflie_logs/avoidance_policy.csv")
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--height-m", type=float, default=0.45)
    parser.add_argument("--clearance-m", type=float, default=0.45)
    parser.add_argument("--hard-clearance-m", type=float, default=0.10)
    parser.add_argument("--max-speed-m-s", type=float, default=0.25)
    parser.add_argument("--target-direction-deg", type=float, default=0.0)
    parser.add_argument("--target-speed-m-s", type=float, default=0.18)
    parser.add_argument("--target-slowdown-gain", type=float, default=0.85)
    parser.add_argument("--target-avoidance-gain", type=float, default=1.0)
    parser.add_argument("--ttc-horizon-s", type=float, default=0.0)
    parser.add_argument("--ttc-hard-s", type=float, default=0.12)
    parser.add_argument("--ttc-gain", type=float, default=1.0)
    parser.add_argument("--range-rate-alpha", type=float, default=0.65)
    parser.add_argument("--range-rate-max-m-s", type=float, default=5.0)
    parser.add_argument("--log-timeout-s", type=float, default=0.5)
    parser.add_argument("--max-vertical-speed-m-s", type=float, default=0.18)
    parser.add_argument("--smoothing-alpha", type=float, default=0.35)
    parser.add_argument("--max-speed-step-m-s", type=float, default=0.03)
    parser.add_argument("--max-yawrate-step-deg-s", type=float, default=6.0)
    parser.add_argument("--max-zdistance-step-m", type=float, default=0.04)
    parser.add_argument("--emergency-clearance-m", type=float, default=0.25)
    parser.add_argument("--emergency-ttc-s", type=float, default=0.0)
    parser.add_argument("--emergency-max-speed-m-s", type=float, default=0.14)
    parser.add_argument("--emergency-speed-step-m-s", type=float, default=0.06)
    parser.add_argument("--absolute-max-speed-m-s", type=float, default=0.35)
    parser.add_argument("--emergency-hold-steps", type=int, default=0)
    parser.add_argument("--anti-oscillation-hold-s", type=float, default=0.0)
    parser.add_argument("--anti-oscillation-min-speed-m-s", type=float, default=0.12)
    parser.add_argument("--anti-oscillation-hard-clearance-m", type=float, default=0.11)
    parser.add_argument("--anti-oscillation-hard-ttc-s", type=float, default=0.25)
    parser.add_argument("--close-escape-clearance-m", type=float, default=0.0)
    parser.add_argument("--close-escape-min-speed-m-s", type=float, default=0.0)
    parser.add_argument("--close-escape-brake-gain", type=float, default=0.0)
    parser.add_argument("--close-escape-brake-max-m-s", type=float, default=0.0)
    parser.add_argument("--height-error-abort-m", type=float, default=0.35, help="Target-relative height abort. Set 0 to disable for vertical-clearance experiments.")
    parser.add_argument("--min-state-height-m", type=float, default=0.10)
    parser.add_argument("--max-state-height-m", type=float, default=1.20)
    parser.add_argument("--vertical-controller", choices=("height", "clearance"), default="height")
    parser.add_argument("--lock-height", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confirm-flight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--approval-manifest", default="artifacts/replay/sim2real_checkpoint_manifest_current_2026-05-20.json")
    args = parser.parse_args()

    if args.controller in ("policy", "ttc-policy") and not args.checkpoint:
        raise SystemExit(f"--checkpoint is required when --controller {args.controller}")
    if args.dry_run:
        model = load_policy(args.checkpoint, args.controller) if args.controller in ("policy", "ttc-policy") else None
        shadow_model = load_policy(args.shadow_checkpoint) if args.shadow_checkpoint else None
        target_shadow_model = load_target_policy(args.target_shadow_checkpoint) if args.target_shadow_checkpoint else None
        ttc_shadow_model = load_ttc_policy(args.ttc_shadow_checkpoint) if args.ttc_shadow_checkpoint else None
        reading = reading_from_telemetry({"range.front": 250.0, "range.back": 2000.0, "range.zrange": args.height_m * 1000.0})
        range_rate = RangerReading(front_m=-0.6, back_m=0.0, left_m=0.0, right_m=0.0, up_m=0.0, zrange_m=0.0)
        command = build_control_command(model, reading, range_rate, args)
        command, emergency = maybe_emergency_command(command, reading, range_rate, args)
        smoothed = smooth_avoidance_command(command, AvoidanceCommand(0.0, 0.0, 0.0, args.height_m), args, emergency=emergency)
        shadow = build_shadow_command(shadow_model, reading, args) if shadow_model else None
        target_shadow = build_target_shadow_command(target_shadow_model, reading, args) if target_shadow_model else None
        ttc_shadow = build_ttc_shadow_command(ttc_shadow_model, reading, range_rate, args) if ttc_shadow_model else None
        print(f"dry_run avoidance command: raw={command} smoothed={smoothed} shadow={shadow} target_shadow={target_shadow} ttc_shadow={ttc_shadow}")
        return
    if not args.confirm_flight:
        raise SystemExit("--confirm-flight is required for real drone control")
    if args.controller in ("policy", "ttc-policy"):
        require_policy_approval(args.checkpoint, args.approval_manifest)
    model = load_policy(args.checkpoint, args.controller) if args.controller in ("policy", "ttc-policy") else None
    shadow_model = load_policy(args.shadow_checkpoint) if args.shadow_checkpoint else None
    target_shadow_model = load_target_policy(args.target_shadow_checkpoint) if args.target_shadow_checkpoint else None
    ttc_shadow_model = load_ttc_policy(args.ttc_shadow_checkpoint) if args.ttc_shadow_checkpoint else None
    rows = run_live(model, shadow_model, target_shadow_model, ttc_shadow_model, args)
    write_rows(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")

if __name__ == "__main__":
    main()
