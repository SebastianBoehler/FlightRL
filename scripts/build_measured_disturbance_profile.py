from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from flightrl.hardware.direct_raw_gate import (
    DirectRawGateThresholds,
    evaluate_direct_raw_replay,
    horizontal_speed_m_s,
    precontact_row,
    value,
)
from flightrl.sixdof.transfer_selection import split_label_path


OUTLIER_FACTOR = 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate a six-DoF disturbance profile from raw live drift logs.")
    parser.add_argument("--live-log", action="append", default=[], required=True, help="LABEL:CSV")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stress-output")
    args = parser.parse_args()

    cases = [split_label_path(item) for item in args.live_log]
    records = [measure_case(label, path) for label, path in cases]
    report = build_report(records)
    write_report(report, Path(args.output))
    if args.stress_output:
        write_report(stress_report(report), Path(args.stress_output))
    print(f"measured_disturbance_profile={args.output}")
    if args.stress_output:
        print(f"stress_disturbance_profile={args.stress_output}")
    print(f"nominal_logs={len(report['summary']['nominal_logs'])}/{len(records)}")


def measure_case(label: str, path: str) -> dict[str, Any]:
    rows = load_rows(path)
    thresholds = DirectRawGateThresholds(min_safe_rows=0, require_commander_pitch_sign=False)
    source = evaluate_direct_raw_replay(rows, thresholds)
    precontact = [row for row in rows if precontact_row(row, thresholds)]
    if len(precontact) < 2:
        raise ValueError(f"{label} has fewer than two pre-contact rows")

    start = precontact[0]
    peak = max(precontact, key=horizontal_speed_m_s)
    start_t = value(start, "host_time_s")
    peak_t = value(peak, "host_time_s")
    elapsed = peak_t - start_t
    if elapsed <= 0.0:
        raise ValueError(f"{label} pre-contact peak does not occur after start")

    start_speed = horizontal_speed_m_s(start)
    peak_speed = horizontal_speed_m_s(peak)
    xy_accel = max(0.0, (peak_speed - start_speed) / elapsed)
    z_accel = (value(peak, "stateEstimate.vz") - value(start, "stateEstimate.vz")) / elapsed
    return {
        "label": label,
        "path": path,
        "rows": len(rows),
        "precontact_rows": len(precontact),
        "start_time_s": start_t,
        "peak_time_s": peak_t,
        "elapsed_s": elapsed,
        "start_horizontal_speed_m_s": start_speed,
        "peak_horizontal_speed_m_s": peak_speed,
        "equivalent_xy_accel_m_s2": xy_accel,
        "equivalent_z_accel_m_s2": z_accel,
        "source_failures": source["failures"],
        "source": source["source"],
    }


def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    xy = np.asarray([record["equivalent_xy_accel_m_s2"] for record in records], dtype=np.float32)
    median_xy = float(np.median(xy))
    threshold = median_xy * OUTLIER_FACTOR
    nominal = [record for record in records if record["equivalent_xy_accel_m_s2"] <= threshold]
    stress = [record for record in records if record not in nominal]
    if not nominal:
        raise ValueError("no nominal disturbance records remain after stress filtering")

    xy_nominal = [record["equivalent_xy_accel_m_s2"] for record in nominal]
    z_nominal = [record["equivalent_z_accel_m_s2"] for record in nominal]
    for record in records:
        record["nominal_profile_source"] = record in nominal

    return {
        "disturbance_profile": {
            "name": "raw_live_drift_measured",
            "world_accel_xy_m_s2": [float(min(xy_nominal)), float(max(xy_nominal))],
            "world_accel_z_m_s2": [float(min(z_nominal)), float(max(z_nominal))],
        },
        "stress_disturbance_profile": stress_profile(stress),
        "summary": {
            "logs": len(records),
            "nominal_logs": [record["label"] for record in nominal],
            "stress_logs": [record["label"] for record in stress],
            "median_equivalent_xy_accel_m_s2": median_xy,
            "stress_cutoff_xy_accel_m_s2": threshold,
            "stress_xy_accel_max_m_s2": float(np.max(xy)),
            "estimator": "precontact horizontal speed slope; logs above 2x median are retained as stress evidence",
        },
        "records": records,
        "safety": "Offline disturbance-profile calibration only; this artifact does not approve live hardware deployment.",
    }


def stress_profile(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    xy = [record["equivalent_xy_accel_m_s2"] for record in records]
    z = [record["equivalent_z_accel_m_s2"] for record in records]
    return {
        "name": "raw_live_drift_stress",
        "world_accel_xy_m_s2": [float(min(xy)), float(max(xy))],
        "world_accel_z_m_s2": [float(min(z)), float(max(z))],
    }


def stress_report(report: dict[str, Any]) -> dict[str, Any]:
    profile = report.get("stress_disturbance_profile")
    if profile is None:
        raise ValueError("no stress records are available for --stress-output")
    records = [record for record in report["records"] if not record["nominal_profile_source"]]
    return {
        "disturbance_profile": profile,
        "summary": {
            "logs": len(records),
            "stress_logs": [record["label"] for record in records],
            "estimator": report["summary"]["estimator"],
        },
        "records": records,
        "safety": report["safety"],
    }


def load_rows(path: str | Path) -> list[dict[str, float]]:
    rows = []
    latest: dict[str, float] = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            latest.update({key: parse_float(raw) for key, raw in row.items() if raw != ""})
            rows.append(dict(latest))
    rows.sort(key=lambda row: value(row, "host_time_s"))
    return rows


def parse_float(raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")


def render_markdown(report: dict[str, Any]) -> str:
    profile = report["disturbance_profile"]
    summary = report["summary"]
    lines = [
        "# Measured Raw Live Drift Disturbance Profile",
        "",
        f"- XY accel range: `{profile['world_accel_xy_m_s2'][0]:.4f}` to `{profile['world_accel_xy_m_s2'][1]:.4f}` m/s^2",
        f"- Z accel range: `{profile['world_accel_z_m_s2'][0]:.4f}` to `{profile['world_accel_z_m_s2'][1]:.4f}` m/s^2",
        f"- Nominal logs: `{', '.join(summary.get('nominal_logs', [])) or 'none'}`",
        f"- Stress logs retained outside nominal profile: `{', '.join(summary.get('stress_logs', [])) or 'none'}`",
        "",
        "| log | nominal | peak speed | equivalent xy accel | equivalent z accel | source failures |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in report["records"]:
        failures = ", ".join(record["source_failures"]) or "none"
        lines.append(
            f"| {record['label']} | `{record['nominal_profile_source']}` | "
            f"{record['peak_horizontal_speed_m_s']:.4f} | {record['equivalent_xy_accel_m_s2']:.4f} | "
            f"{record['equivalent_z_accel_m_s2']:.4f} | {failures} |"
        )
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
