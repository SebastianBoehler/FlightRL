from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.sixdof.readiness import compact_puffer_export, compact_residual_sweep, compact_training_throughput, format_optional, format_task_gates, puffer_export_failures, read_json, summary, training_throughput_failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote 6-DoF checkpoint candidates using sim, edge, room, and native parity evidence")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--room-report", default=None)
    parser.add_argument("--native-parity", default=None)
    parser.add_argument("--profile-matrix", default=None, help="Optional build_sixdof_profile_matrix.py JSON report.")
    parser.add_argument("--replay-comparison", default=None, help="Optional compare_crazyflie_replay.py JSON report with aligned real/sim signals.")
    parser.add_argument("--residual-sweep", default=None, help="Optional run_sixdof_residual_ppo_sweep.py JSON report.")
    parser.add_argument("--training-throughput", default=None, help="Optional benchmark_sixdof_training_throughput.py JSON report.")
    parser.add_argument("--puffer-export", default=None, help="Optional build_sixdof_puffer_export_report.py JSON report.")
    parser.add_argument("--require-replay-comparison", action="store_true")
    parser.add_argument("--require-training-throughput", action="store_true")
    parser.add_argument("--require-puffer-export", action="store_true")
    parser.add_argument("--output", default="artifacts/replay/sixdof_readiness_report.json")
    parser.add_argument("--max-latency-us", type=float, default=50.0)
    parser.add_argument("--max-native-state-rmse", type=float, default=1e-5)
    parser.add_argument("--max-native-range-rmse", type=float, default=1.0)
    parser.add_argument("--max-replay-state-rmse", type=float, default=0.5)
    parser.add_argument("--max-replay-range-rmse-mm", type=float, default=300.0)
    parser.add_argument("--min-replay-overlap-s", type=float, default=1.0)
    parser.add_argument("--min-training-total-sps", type=float, default=0.0)
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
    profile = read_json(args.profile_matrix) if args.profile_matrix else None
    replay = read_json(args.replay_comparison) if args.replay_comparison else None
    residual = read_json(args.residual_sweep) if args.residual_sweep else None
    throughput = read_json(args.training_throughput) if args.training_throughput else None
    puffer = read_json(args.puffer_export) if args.puffer_export else None
    global_evidence = {
        "room": compact_room(room),
        "native_parity": compact_native_parity(native, args.max_native_state_rmse, args.max_native_range_rmse),
        "profile_matrix": compact_profile_matrix(profile),
        "replay_comparison": compact_replay_comparison(replay, args),
        "residual_sweep": compact_residual_sweep(residual),
        "training_throughput": compact_training_throughput(throughput),
        "puffer_export": compact_puffer_export(puffer),
    }
    records = [evaluate_record(record, global_evidence, args.max_latency_us, args.require_training_throughput, args.min_training_total_sps, args.require_puffer_export) for record in readiness_candidates(matrix)]
    return {
        "matrix": args.matrix,
        "room_report": args.room_report,
        "native_parity": args.native_parity,
        "profile_matrix": args.profile_matrix,
        "replay_comparison": args.replay_comparison,
        "residual_sweep": args.residual_sweep,
        "training_throughput": args.training_throughput,
        "puffer_export": args.puffer_export,
        "thresholds": {
            "max_latency_us": args.max_latency_us,
            "max_native_state_rmse": args.max_native_state_rmse,
            "max_native_range_rmse": args.max_native_range_rmse,
            "max_replay_state_rmse": args.max_replay_state_rmse,
            "max_replay_range_rmse_mm": args.max_replay_range_rmse_mm,
            "min_replay_overlap_s": args.min_replay_overlap_s,
            "require_training_throughput": args.require_training_throughput,
            "min_training_total_sps": args.min_training_total_sps,
            "require_puffer_export": args.require_puffer_export,
        },
        "global_evidence": global_evidence,
        "records": records,
        "summary": summary(records),
        "safety": "Readiness is for simulation/edge bench promotion only; it is not approval for autonomous live flight.",
    }


def evaluate_record(task_and_record: tuple[str, dict], global_evidence: dict, max_latency_us: float, require_training_throughput: bool = False, min_training_total_sps: float = 0.0, require_puffer_export: bool = False) -> dict:
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
    profile = profile_record(record, global_evidence.get("profile_matrix", {"present": False}))
    if global_evidence.get("profile_matrix", {}).get("present") and "position_yaw" in record.get("tasks", [task]) and not profile.get("present"):
        failures.append("profile_matrix_missing")
    elif profile.get("present") and not profile.get("passed_all_profiles", False):
        failures.append("profile_matrix")
    replay = global_evidence["replay_comparison"]
    if replay.get("required") and not replay.get("present"):
        failures.append("replay_comparison_missing")
    elif replay.get("present") and not replay.get("passed"):
        failures.append("replay_comparison")
    failures.extend(training_throughput_failures(global_evidence.get("training_throughput", {}), require=require_training_throughput, min_total_sps=min_training_total_sps))
    failures.extend(puffer_export_failures(global_evidence.get("puffer_export", {}), require=require_puffer_export))
    return {
        "task": task,
        "label": record["label"],
        "checkpoint": record["checkpoint"],
        "tasks": record.get("tasks", [task]),
        "ready": not failures,
        "failures": failures,
        "sim": {
            "passed": record.get("passed", False),
            "mean_completed_fraction": record.get("mean_completed_fraction"),
            "mean_position_error_m": record.get("mean_position_error_m"),
            "mean_yaw_error_rad": record.get("mean_yaw_error_rad"),
            "yaw_error_p95_rad": record.get("yaw_error_p95_rad"),
            "clearance_p01_m": record.get("clearance_p01_m"),
            "per_task_gate": record.get("per_task_gate", {}),
        },
        "edge_parity": parity,
        "edge_latency": latency,
        "profile_matrix": profile,
    }


def readiness_candidates(matrix: dict) -> list[tuple[str, dict]]:
    candidates = sorted(matrix.get("best_by_task", {}).items())
    multitask = matrix.get("best_multitask")
    if multitask:
        candidates.append(("multitask", multitask))
    return candidates


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


def compact_profile_matrix(report: dict | None) -> dict:
    if not report:
        return {"present": False, "profiles": [], "by_checkpoint": {}, "by_label": {}}
    records = [compact_profile_record(record) for record in report.get("records", [])]
    return {
        "present": True,
        "profiles": report.get("profiles", []),
        "by_checkpoint": {record["checkpoint"]: record for record in records},
        "by_label": {record["label"]: record for record in records},
    }
def compact_profile_record(record: dict) -> dict:
    return {
        "present": True, "label": record["label"], "checkpoint": record["checkpoint"],
        "passed_all_profiles": record["passed_all_profiles"], "missing_profiles": record["missing_profiles"],
        "failures_by_profile": record["failures_by_profile"], "worst_survival_fraction": record["worst_survival_fraction"],
        "worst_completed_fraction": record["worst_completed_fraction"], "worst_position_error_m": record["worst_position_error_m"],
        "worst_clearance_p01_m": record["worst_clearance_p01_m"],
        "worst_yaw_error_rad": record.get("worst_yaw_error_rad"),
    }

def profile_record(record: dict, profile_matrix: dict) -> dict:
    if not profile_matrix.get("present"):
        return {"present": False}
    return profile_matrix["by_checkpoint"].get(record["checkpoint"]) or profile_matrix["by_label"].get(record["label"]) or {"present": False}

def compact_replay_comparison(report: dict | None, args: argparse.Namespace) -> dict:
    if not report:
        return {"present": False, "required": args.require_replay_comparison, "passed": not args.require_replay_comparison}
    aligned = report.get("aligned", {})
    signals = aligned.get("signals", {})
    worst_state = worst_rmse(signals, "stateEstimate.")
    worst_range = worst_rmse(signals, "range.")
    failures = []
    if aligned.get("overlap_duration_s", 0.0) < args.min_replay_overlap_s:
        failures.append("overlap")
    if worst_state is None or worst_state > args.max_replay_state_rmse:
        failures.append("state_rmse")
    if worst_range is None or worst_range > args.max_replay_range_rmse_mm:
        failures.append("range_rmse")
    return {
        "present": True,
        "required": args.require_replay_comparison,
        "passed": not failures,
        "failures": failures,
        "samples": aligned.get("samples", 0),
        "overlap_duration_s": aligned.get("overlap_duration_s", 0.0),
        "worst_state_rmse": worst_state,
        "worst_range_rmse_mm": worst_range,
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


def render_markdown(report: dict) -> str:
    lines = [
        "# 6-DoF Readiness Report",
        "",
        "| scope | tasks | label | ready | failures | latency us | completed | pos err m | clearance p01 m |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for record in report["records"]:
        latency = record["edge_latency"].get("per_sample_us")
        lines.append(
            f"| {record['task']} | {', '.join(record['tasks'])} | {record['label']} | {record['ready']} | {', '.join(record['failures']) or 'none'} | "
            f"{format_optional(latency)} | {record['sim']['mean_completed_fraction']:.4f} | "
            f"{record['sim']['mean_position_error_m']:.4f} | {record['sim']['clearance_p01_m']:.4f} |"
        )
    room = report["global_evidence"]["room"]
    native = report["global_evidence"]["native_parity"]
    profile = report["global_evidence"]["profile_matrix"]
    replay = report["global_evidence"]["replay_comparison"]
    if any(record["sim"].get("per_task_gate") for record in report["records"]):
        lines.extend(["", "## Per-Task Gates", ""])
        for record in report["records"]:
            gates = record["sim"].get("per_task_gate", {})
            if gates:
                lines.append(f"- `{record['label']}`: {format_task_gates(gates)}")
    lines.extend(
        [
            "",
            f"Room ready: `{room['mapping_ready']}`; points=`{room.get('point_count')}`; warnings=`{', '.join(room.get('warnings', [])) or 'none'}`.",
            f"Native parity: `{native['passed']}`; worst_state_rmse=`{native.get('worst_state_rmse')}`; worst_range_rmse=`{native.get('worst_range_rmse')}`.",
            f"Profile matrix: present=`{profile['present']}`; profiles=`{', '.join(profile.get('profiles', [])) or 'none'}`.",
            f"Replay comparison: `{replay['passed']}`; present=`{replay.get('present')}`; worst_state_rmse=`{replay.get('worst_state_rmse')}`; worst_range_rmse_mm=`{replay.get('worst_range_rmse_mm')}`.",
            f"Residual sweep: present=`{report['global_evidence']['residual_sweep']['present']}`; best=`{(report['global_evidence']['residual_sweep'].get('best') or {}).get('name')}`.",
            f"Training throughput: present=`{report['global_evidence']['training_throughput']['present']}`; best_total_sps=`{(report['global_evidence']['training_throughput'].get('best_total_sps') or {}).get('total_sps')}`.",
            f"Puffer export: present=`{report['global_evidence']['puffer_export']['present']}`; passed=`{report['global_evidence']['puffer_export'].get('passed')}`; env=`{report['global_evidence']['puffer_export'].get('env_name')}`.",
            "",
            report["safety"],
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
