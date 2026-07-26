from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from flightrl.hardware.avoidance_policy import command_array, reactive_clearance_command, reading_from_telemetry
from flightrl.hardware.ttc_policy import command_from_ttc_model, load_ttc_policy, rate_from_telemetry
from flightrl.hardware.ttc_shadow import GROUPS, group_metrics, groups_for_row, write_ttc_shadow_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a TTC policy with the TTC teacher on live telemetry rows.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height-m", type=float, default=0.50)
    parser.add_argument("--clearance-m", type=float, default=0.45)
    parser.add_argument("--hard-clearance-m", type=float, default=0.08)
    parser.add_argument("--max-speed-m-s", type=float, default=0.65)
    parser.add_argument("--ttc-horizon-s", type=float, default=0.75)
    parser.add_argument("--ttc-hard-s", type=float, default=0.15)
    parser.add_argument("--ttc-gain", type=float, default=1.1)
    args = parser.parse_args()

    model = load_ttc_policy(args.checkpoint)
    grouped: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {name: [] for name in GROUPS}
    latest: dict[str, float] = {}
    for raw in csv.DictReader(Path(args.input).open()):
        latest.update({key: float(value) for key, value in raw.items() if numeric(value)})
        if not has_ttc_inputs(latest):
            continue
        reading = reading_from_telemetry(latest)
        rate = rate_from_telemetry(latest)
        teacher = reactive_clearance_command(
            reading,
            range_rate_m_s=rate,
            clearance_m=args.clearance_m,
            hard_clearance_m=args.hard_clearance_m,
            target_height_m=args.height_m,
            max_speed_m_s=args.max_speed_m_s,
            ttc_horizon_s=args.ttc_horizon_s,
            ttc_hard_s=args.ttc_hard_s,
            ttc_gain=args.ttc_gain,
        )
        predicted = command_from_ttc_model(model, reading, rate, max_speed_m_s=args.max_speed_m_s)
        for group in groups_for_row(latest):
            grouped[group].append((command_array(teacher), command_array(predicted)))
    report = {
        "input": args.input,
        "target": "ttc_teacher",
        "shadow_prefix": "ttc_policy",
        "groups": {name: group_metrics(pairs) for name, pairs in grouped.items()},
    }
    write_ttc_shadow_report(report, args.output)
    print(f"ttc_teacher_shadow_report={args.output}")
    print(f"samples={report['groups']['all']['samples']}")


def has_ttc_inputs(row: dict[str, float]) -> bool:
    return all(
        key in row
        for key in (
            "range.front",
            "range.back",
            "range.left",
            "range.right",
            "range.zrange",
            "range_rate_front_m_s",
            "range_rate_back_m_s",
            "range_rate_left_m_s",
            "range_rate_right_m_s",
        )
    )


def numeric(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


if __name__ == "__main__":
    main()
