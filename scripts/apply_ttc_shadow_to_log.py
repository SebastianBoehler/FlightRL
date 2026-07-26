from __future__ import annotations

import argparse
import csv
from pathlib import Path

from flightrl.hardware.avoidance_policy import reading_from_telemetry
from flightrl.hardware.avoidance_shadow import shadow_command_row
from flightrl.hardware.ttc_policy import command_from_ttc_model, load_ttc_policy, rate_from_telemetry


def main() -> None:
    parser = argparse.ArgumentParser(description="Append TTC policy shadow commands to a Crazyflie CSV log.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix", default="ttc_shadow")
    parser.add_argument("--max-speed-m-s", type=float, default=0.65)
    args = parser.parse_args()

    model = load_ttc_policy(args.checkpoint)
    rows = shadow_rows(model, Path(args.input), args)
    write_rows(Path(args.output), rows)
    print(f"ttc_shadow_log={args.output}")
    print(f"rows={len(rows)}")


def shadow_rows(model, input_csv: Path, args) -> list[dict[str, float]]:
    latest: dict[str, float] = {}
    rows: list[dict[str, float]] = []
    for raw in csv.DictReader(input_csv.open()):
        latest.update({key: float(value) for key, value in raw.items() if numeric(value)})
        row = dict(latest)
        if has_ttc_inputs(row):
            command = command_from_ttc_model(
                model,
                reading_from_telemetry(row),
                rate_from_telemetry(row),
                max_speed_m_s=args.max_speed_m_s,
            )
            row.update(shadow_command_row(command, prefix=args.prefix))
        rows.append(row)
    return rows


def has_ttc_inputs(row: dict[str, float]) -> bool:
    required = (
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
    return all(key in row for key in required)


def write_rows(output: Path, rows: list[dict[str, float]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["host_time_s"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def numeric(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


if __name__ == "__main__":
    main()
