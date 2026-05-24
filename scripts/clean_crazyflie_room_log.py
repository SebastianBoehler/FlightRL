from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.hardware.room_log_cleaner import clean_room_rows, load_csv_rows, write_csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove impossible estimator speed spikes from Crazyflie room logs")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--max-step-speed-m-s", type=float, required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output = Path(args.output or input_path.with_name(f"{input_path.stem}.clean.csv"))
    report_path = Path(args.report or output.with_suffix(".clean.json"))

    rows, fieldnames = load_csv_rows(input_path)
    result = clean_room_rows(rows, max_step_speed_m_s=args.max_step_speed_m_s)
    write_csv_rows(output, result.rows, fieldnames)
    report = {
        "input": str(input_path),
        "output": str(output),
        "max_step_speed_m_s": args.max_step_speed_m_s,
        "input_count": result.input_count,
        "kept_count": result.kept_count,
        "dropped_count": result.dropped_count,
        "dropped_fraction": result.dropped_fraction,
        "max_observed_step_speed_m_s": result.max_observed_step_speed_m_s,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"wrote {output} and {report_path}; "
        f"dropped={result.dropped_count}/{result.input_count} ({result.dropped_fraction:.4f})"
    )


if __name__ == "__main__":
    main()
