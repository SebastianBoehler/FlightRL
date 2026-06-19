from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.sim2real.hover_yaw_stability import summarize_hover_yaw_logs, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Crazyflie firmware hover/yaw stability logs")
    parser.add_argument("--log", action="append", type=Path, required=True, help="Clean hover/yaw CSV log. Repeatable.")
    parser.add_argument("--contaminated-log", action="append", type=Path, default=[], help="Known collision/contaminated log. Repeatable.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/replay/firmware_hover_yaw_stability.json"))
    parser.add_argument("--stable-after-s", type=float, default=1.0)
    parser.add_argument("--max-xy-span-m", type=float, default=0.18)
    parser.add_argument("--max-z-span-m", type=float, default=0.08)
    parser.add_argument("--min-side-range-mm", type=float, default=450.0)
    parser.add_argument("--min-battery-v", type=float, default=3.7)
    args = parser.parse_args()

    report = summarize_hover_yaw_logs(
        args.log,
        contaminated_logs=args.contaminated_log,
        stable_after_s=args.stable_after_s,
        max_xy_span_m=args.max_xy_span_m,
        max_z_span_m=args.max_z_span_m,
        min_side_range_mm=args.min_side_range_mm,
        min_battery_v=args.min_battery_v,
    )
    write_report(report, args.output)
    print(f"summary={args.output}")
    print(f"markdown={args.output.with_suffix('.md')}")
    print(f"stability_ready={report['summary']['stability_ready']}")


if __name__ == "__main__":
    main()
