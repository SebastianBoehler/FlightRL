from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from flightrl.hardware.calibration_quality import summarize_calibration_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Crazyflie calibration-flight replay quality")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument("--min-duration-s", type=float, default=8.0)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--min-floor-valid-ratio", type=float, default=0.5)
    parser.add_argument("--min-yaw-span-deg", type=float, default=45.0)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output = Path(args.output or input_path.with_suffix(".calibration.json"))
    markdown = Path(args.markdown or output.with_suffix(".md"))
    with input_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = summarize_calibration_log(
        rows,
        min_duration_s=args.min_duration_s,
        min_rows=args.min_rows,
        min_floor_valid_ratio=args.min_floor_valid_ratio,
        min_yaw_span_deg=args.min_yaw_span_deg,
    )
    report = {"input": str(input_path), "summary": summary}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    markdown.write_text(render_markdown(report))
    print(f"wrote {output} and {markdown}; replay_calibration_ready={summary['replay_calibration_ready']}")
    if args.strict and not summary["replay_calibration_ready"]:
        raise SystemExit(1)


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    rows = [
        ("input", report["input"]),
        ("replay_calibration_ready", str(summary["replay_calibration_ready"])),
        ("failures", ", ".join(summary["failures"]) or "none"),
        ("rows", str(summary["rows"])),
        ("duration_s", f"{summary['duration_s']:.2f}"),
        ("sample_rate_hz", f"{summary['sample_rate_hz']:.2f}"),
        ("time_monotonic", str(summary["time_monotonic"])),
        ("missing_columns", ", ".join(summary["missing_columns"]) or "none"),
        ("floor_valid_ratio", f"{summary['floor_valid_ratio']:.3f}"),
        ("yaw_span_deg", f"{summary['yaw_span_deg']:.1f}"),
        ("xy_span_m", f"{summary['xy_span_m']:.3f}"),
        ("z_span_m", f"{summary['z_span_m']:.3f}"),
        ("modes", ", ".join(summary["modes"]) or "none"),
    ]
    table = ["# Crazyflie Calibration Flight Quality", "", "| metric | value |", "| --- | --- |"]
    table.extend(f"| {key} | {value} |" for key, value in rows)
    return "\n".join(table) + "\n"


if __name__ == "__main__":
    main()
