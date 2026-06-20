from __future__ import annotations

import argparse

from flightrl.hardware.ttc_shadow import evaluate_ttc_shadow_log, write_ttc_shadow_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TTC shadow action gaps from a Crazyflie avoidance log")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", choices=("raw", "held"), default="raw")
    parser.add_argument("--shadow-prefix", default="ttc_shadow")
    args = parser.parse_args()

    report = evaluate_ttc_shadow_log(args.input, target=args.target, shadow_prefix=args.shadow_prefix)
    write_ttc_shadow_report(report, args.output)
    print(f"ttc_shadow_report={args.output}")
    print(f"samples={report['groups']['all']['samples']}")


if __name__ == "__main__":
    main()
