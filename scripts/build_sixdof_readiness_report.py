from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote 6-DoF checkpoint candidates using sim, edge, room, and native parity evidence")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--room-report", default=None)
    parser.add_argument("--native-parity", default=None)
    parser.add_argument("--output", default="artifacts/replay/sixdof_readiness_report.json")
    parser.add_argument("--max-latency-us", type=float, default=50.0)
    parser.add_argument("--max-native-state-rmse", type=float, default=1e-5)
    parser.add_argument("--max-native-range-rmse", type=float, default=1.0)
    args = parser.parse_args()

    report = build_report(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"readiness={output}")
    print(f"markdown={output.with_suffix('.md')}")


def build_report(args: argparse.Namespace) -> dict:
    matrix = read_json(args.matrix)
    room = read_json(args.room_report) if args.room_report else None
    native = read_json(args.native_parity) if args.native_parity else None
    global_evidence = {
        "room": compact_room(room),
        "native_parity": compact_native_parity(native, args.max_native_state_rmse, args.max_native_range_rmse),
    }
    records = [
        evaluate_record(record, global_evidence, args.max_latency_us)
        for record in sorted(matrix.get("best_by_task", {}).items())
    ]
    return {
        "matrix": args.matrix,
        "room_report": args.room_report,
        "native_parity": args.native_parity,
        "thresholds": {
            "max_latency_us": args.max_latency_us,
            "max_native_state_rmse": args.max_native_state_rmse,
            "max_native_range_rmse": args.max_native_range_rmse,
        },
        "global_evidence": global_evidence,
        "records": records,
        "summary": summary(records),
        "safety": "Readiness is for simulation/edge bench promotion only; it is not approval for autonomous live flight.",
    }


def evaluate_record(task_and_record: tuple[str, dict], global_evidence: dict, max_latency_us: float) -> dict:
    task, record = task_and_record
    failures = []
    if not record.get("passed", False):
        failures.append("sim_gate")
    parity = record.get("edge_parity", {})
    if not parity.get("passed", False):
        failures.append("edge_parity")
    latency = record.get("edge_latency", {})
    latency_us = latency.get("per_sample_us")
    if latency_us is None:
        failures.append("edge_latency_missing")
    elif latency_us > max_latency_us:
        failures.append("edge_latency_slow")
    if not global_evidence["room"]["mapping_ready"]:
        failures.append("room_map")
    if not global_evidence["native_parity"]["passed"]:
        failures.append("native_parity")
    return {
        "task": task,
        "label": record["label"],
        "checkpoint": record["checkpoint"],
        "ready": not failures,
        "failures": failures,
        "sim": {
            "passed": record.get("passed", False),
            "mean_completed_fraction": record.get("mean_completed_fraction"),
            "mean_position_error_m": record.get("mean_position_error_m"),
            "mean_yaw_error_rad": record.get("mean_yaw_error_rad"),
            "yaw_error_p95_rad": record.get("yaw_error_p95_rad"),
            "clearance_p01_m": record.get("clearance_p01_m"),
        },
        "edge_parity": parity,
        "edge_latency": latency,
    }


def compact_room(report: dict | None) -> dict:
    if not report:
        return {"present": False, "mapping_ready": False}
    summary = report.get("summary", {})
    estimate = report.get("room_estimate", {})
    return {
        "present": True,
        "mapping_ready": bool(summary.get("mapping_ready", False)),
        "failures": summary.get("failures", []),
        "point_count": summary.get("point_count"),
        "duration_s": summary.get("duration_s"),
        "width_m": estimate.get("width_m"),
        "depth_m": estimate.get("depth_m"),
        "height_m": estimate.get("height_m"),
        "warnings": estimate.get("warnings", []),
    }


def compact_native_parity(report: dict | None, max_state_rmse: float, max_range_rmse: float) -> dict:
    if not report:
        return {"present": False, "passed": False, "failures": ["missing"]}
    signals = report.get("aligned", {}).get("signals", {})
    failures = []
    worst_state = worst_rmse(signals, "stateEstimate.")
    worst_range = worst_rmse(signals, "range.")
    if worst_state is None or worst_state > max_state_rmse:
        failures.append("state_rmse")
    if worst_range is None or worst_range > max_range_rmse:
        failures.append("range_rmse")
    mismatches = native_mismatch_count(report)
    if mismatches:
        failures.append("termination_mismatch")
    return {
        "present": True,
        "passed": not failures,
        "failures": failures,
        "samples": report.get("aligned", {}).get("samples", 0),
        "overlap_duration_s": report.get("aligned", {}).get("overlap_duration_s", 0.0),
        "worst_state_rmse": worst_state,
        "worst_range_rmse": worst_range,
        "termination_mismatches": mismatches,
    }


def worst_rmse(signals: dict, prefix: str) -> float | None:
    values = [metrics["rmse"] for key, metrics in signals.items() if key.startswith(prefix) and "rmse" in metrics]
    return max(values) if values else None


def native_mismatch_count(report: dict) -> int:
    return int(
        sum(
            int(profile.get("terminal_mismatches", 0)) + int(profile.get("truncation_mismatches", 0))
            for profile in report.get("profiles", [])
        )
    )


def summary(records: list[dict]) -> dict:
    ready = [record for record in records if record["ready"]]
    return {"total": len(records), "ready": len(ready), "blocked": len(records) - len(ready), "ready_tasks": [record["task"] for record in ready]}


def read_json(path: str | None) -> dict:
    return json.loads(Path(path).read_text()) if path else {}


def render_markdown(report: dict) -> str:
    lines = [
        "# 6-DoF Readiness Report",
        "",
        "| task | label | ready | failures | latency us | completed | pos err m | clearance p01 m |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for record in report["records"]:
        latency = record["edge_latency"].get("per_sample_us")
        lines.append(
            f"| {record['task']} | {record['label']} | {record['ready']} | {', '.join(record['failures']) or 'none'} | "
            f"{format_optional(latency)} | {record['sim']['mean_completed_fraction']:.4f} | "
            f"{record['sim']['mean_position_error_m']:.4f} | {record['sim']['clearance_p01_m']:.4f} |"
        )
    room = report["global_evidence"]["room"]
    native = report["global_evidence"]["native_parity"]
    lines.extend(
        [
            "",
            f"Room ready: `{room['mapping_ready']}`; points=`{room.get('point_count')}`; warnings=`{', '.join(room.get('warnings', [])) or 'none'}`.",
            f"Native parity: `{native['passed']}`; worst_state_rmse=`{native.get('worst_state_rmse')}`; worst_range_rmse=`{native.get('worst_range_rmse')}`.",
            "",
            report["safety"],
        ]
    )
    return "\n".join(lines)


def format_optional(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


if __name__ == "__main__":
    main()
