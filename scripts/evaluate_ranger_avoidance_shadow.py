from __future__ import annotations

import argparse

from flightrl.hardware.avoidance_shadow import evaluate_shadow_log, write_shadow_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a ranger checkpoint over a Crazyflie avoidance log without hardware control")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-speed-m-s", type=float, default=1.1)
    parser.add_argument("--max-yawrate-deg-s", type=float, default=45.0)
    args = parser.parse_args()

    report = evaluate_shadow_log(
        checkpoint=args.checkpoint,
        input_csv=args.input,
        max_speed_m_s=args.max_speed_m_s,
        max_yawrate_deg_s=args.max_yawrate_deg_s,
    )
    write_shadow_report(report, args.output)
    print(f"shadow_report={args.output}")
    print(f"samples={report['samples']} passed={report['passed']}")


if __name__ == "__main__":
    main()
