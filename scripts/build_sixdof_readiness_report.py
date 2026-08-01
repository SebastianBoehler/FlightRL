from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.evidence_scope import DESKTOP_DEVELOPMENT_SCOPE
from flightrl.evidence_values import exact_true, failure_strings, finite_number
from flightrl.sixdof.candidate_evidence import (
    validate_desktop_identities,
    validate_readiness_candidate,
)
from flightrl.sixdof.native_readiness import compact_native_parity
from flightrl.sixdof.puffer_readiness import (
    compact_puffer_export,
    puffer_export_failures,
)
from flightrl.sixdof.profile_readiness import compact_profile_matrix, profile_record
from flightrl.sixdof.readiness import (
    compact_residual_sweep,
    compact_training_throughput,
    read_json,
    render_markdown,
    summary,
    training_throughput_failures,
)
from flightrl.sixdof.readiness_evidence import compact_replay_comparison, compact_room


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Mac/desktop development readiness from simulation, desktop CPU, room, and native parity evidence")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--room-report", default=None)
    parser.add_argument("--native-parity", default=None)
    parser.add_argument("--profile-matrix", default=None, help="Optional build_sixdof_profile_matrix.py JSON report.")
    parser.add_argument("--replay-comparison", default=None, help="Optional compare_crazyflie_replay.py JSON report with aligned real/sim signals.")
    parser.add_argument(
        "--residual-sweep",
        default=None,
        help="Optional desktop residual-policy sweep JSON report.",
    )
    parser.add_argument("--training-throughput", default=None, help="Optional benchmark_sixdof_training_throughput.py JSON report.")
    parser.add_argument("--puffer-export", default=None, help="Optional build_sixdof_puffer_export_report.py JSON report.")
    parser.add_argument("--require-replay-comparison", action="store_true")
    parser.add_argument("--require-training-throughput", action="store_true")
    parser.add_argument("--require-puffer-export", action="store_true")
    parser.add_argument("--output", default="artifacts/replay/sixdof_readiness_report.json")
    parser.add_argument("--max-desktop-latency-us", type=float, default=50.0)
    parser.add_argument("--max-native-state-rmse", type=float, default=1e-5)
    parser.add_argument("--max-native-range-rmse", type=float, default=1.0)
    parser.add_argument("--max-replay-state-rmse", type=float, default=0.5)
    parser.add_argument("--max-replay-range-rmse-mm", type=float, default=300.0)
    parser.add_argument("--min-replay-overlap-s", type=float, default=1.0)
    parser.add_argument("--min-training-total-sps", type=float, default=0.0)
    args = parser.parse_args()

    try:
        report = build_report(args)
    except ValueError as error:
        parser.error(str(error))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"readiness={output}")
    print(f"markdown={output.with_suffix('.md')}")


def build_report(args: argparse.Namespace) -> dict:
    validate_thresholds(args)
    matrix = read_json(args.matrix)
    validate_matrix(matrix)
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
    records = [
        evaluate_record(
            record,
            global_evidence,
            args.max_desktop_latency_us,
            args.require_training_throughput,
            args.min_training_total_sps,
            args.require_puffer_export,
            require_checkpoint_file=True,
        )
        for record in readiness_candidates(matrix)
    ]
    return {
        "evidence_scope": DESKTOP_DEVELOPMENT_SCOPE,
        "deployment_authority": False,
        "matrix": args.matrix,
        "room_report": args.room_report,
        "native_parity": args.native_parity,
        "profile_matrix": args.profile_matrix,
        "replay_comparison": args.replay_comparison,
        "residual_sweep": args.residual_sweep,
        "training_throughput": args.training_throughput,
        "puffer_export": args.puffer_export,
        "thresholds": {
            "max_desktop_latency_us": args.max_desktop_latency_us,
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
        "safety": "Mac/desktop development readiness only; not AI Deck deployment readiness or autonomous live-flight authority.",
    }


def evaluate_record(
    task_and_record: tuple[str, dict],
    global_evidence: dict,
    max_desktop_latency_us: float,
    require_training_throughput: bool = False,
    min_training_total_sps: float = 0.0,
    require_puffer_export: bool = False,
    require_checkpoint_file: bool = False,
) -> dict:
    task, record = task_and_record
    reject_legacy_edge_evidence(record)
    latency_limit = finite_number(max_desktop_latency_us)
    if latency_limit is None or latency_limit < 0.0:
        raise ValueError("max_desktop_latency_us must be a finite nonnegative number")
    validate_readiness_candidate(record)
    if task != "multitask" and task not in record["tasks"]:
        raise ValueError(f"candidate task key {task!r} is not bound to its task list")
    failures = []
    checkpoint = record["checkpoint"]
    checkpoint_exists = Path(checkpoint).exists()
    if require_checkpoint_file and not checkpoint_exists:
        failures.append("checkpoint_missing")
    elif require_checkpoint_file:
        validate_desktop_identities(record, checkpoint)
    declared_failures = failure_strings(record.get("failures", []))
    if not exact_true(record.get("passed")) or declared_failures != []:
        failures.append("sim_gate")
    parity = record.get("desktop_parity", {})
    if not isinstance(parity, dict) or not exact_true(parity.get("passed")):
        failures.append("desktop_parity")
    latency = record.get("desktop_latency", {})
    latency_us = finite_number(latency.get("per_sample_us")) if isinstance(latency, dict) else None
    latency_sps = finite_number(latency.get("samples_per_second")) if isinstance(latency, dict) else None
    if not exact_true(latency.get("present")) or latency_us is None or latency_us <= 0.0 or latency_sps is None or latency_sps <= 0.0:
        failures.append("desktop_latency_missing")
    elif latency_us > latency_limit:
        failures.append("desktop_latency_slow")
    if not exact_true(global_evidence["room"].get("mapping_ready")):
        failures.append("room_map")
    if not exact_true(global_evidence["native_parity"].get("passed")):
        failures.append("native_parity")
    profile = profile_record(record, global_evidence.get("profile_matrix", {"present": False}))
    if exact_true(global_evidence.get("profile_matrix", {}).get("present")) and "position_yaw" in record.get("tasks", [task]) and not exact_true(profile.get("present")):
        failures.append("profile_matrix_missing")
    elif exact_true(profile.get("present")) and not exact_true(profile.get("passed_all_profiles")):
        failures.append("profile_matrix")
    replay = global_evidence["replay_comparison"]
    if exact_true(replay.get("required")) and not exact_true(replay.get("present")):
        failures.append("replay_comparison_missing")
    elif exact_true(replay.get("present")) and not exact_true(replay.get("passed")):
        failures.append("replay_comparison")
    failures.extend(
        training_throughput_failures(
            global_evidence.get("training_throughput", {}),
            require=require_training_throughput,
            min_total_sps=min_training_total_sps,
            controller=record["controller"],
            tasks=record["tasks"],
        )
    )
    failures.extend(
        puffer_export_failures(
            global_evidence.get("puffer_export", {}),
            require=require_puffer_export,
            candidate=record,
        )
    )
    return {
        "task": task,
        "label": record["label"],
        "checkpoint": checkpoint,
        "checkpoint_exists": checkpoint_exists,
        "tasks": record.get("tasks", [task]),
        "ready": not failures,
        "failures": failures,
        "sim": {
            "passed": exact_true(record.get("passed")) and declared_failures == [],
            "mean_completed_fraction": record.get("mean_completed_fraction"),
            "mean_position_error_m": record.get("mean_position_error_m"),
            "mean_yaw_error_rad": record.get("mean_yaw_error_rad"),
            "yaw_error_p95_rad": record.get("yaw_error_p95_rad"),
            "clearance_p01_m": record.get("clearance_p01_m"),
            "per_task_gate": record.get("per_task_gate", {}),
        },
        "desktop_parity": parity,
        "desktop_latency": latency,
        "profile_matrix": profile,
    }


def reject_legacy_edge_evidence(record: dict) -> None:
    fields = [field for field in ("edge_parity", "edge_latency") if field in record]
    if fields:
        joined = ", ".join(fields)
        raise ValueError(f"legacy {joined} evidence is non-authoritative; regenerate the candidate matrix with desktop_* fields")


def readiness_candidates(matrix: dict) -> list[tuple[str, dict]]:
    candidates = sorted(matrix.get("best_by_task", {}).items())
    multitask = matrix.get("best_multitask")
    if multitask:
        candidates.append(("multitask", multitask))
    return candidates


def validate_matrix(matrix: object) -> None:
    if not isinstance(matrix, dict):
        raise ValueError("candidate matrix must be a JSON object")
    if matrix.get("evidence_scope") != DESKTOP_DEVELOPMENT_SCOPE or matrix.get("deployment_authority") is not False:
        raise ValueError("candidate matrix scope is invalid; regenerate desktop evidence")
    if not isinstance(matrix.get("best_by_task"), dict):
        raise ValueError("candidate matrix best_by_task is missing or invalid")
    multitask = matrix.get("best_multitask")
    if multitask is not None and not isinstance(multitask, dict):
        raise ValueError("candidate matrix best_multitask is invalid")


def validate_thresholds(args: argparse.Namespace) -> None:
    names = (
        "max_desktop_latency_us",
        "max_native_state_rmse",
        "max_native_range_rmse",
        "max_replay_state_rmse",
        "max_replay_range_rmse_mm",
        "min_replay_overlap_s",
        "min_training_total_sps",
    )
    if any((value := finite_number(getattr(args, name))) is None or value < 0.0 for name in names):
        raise ValueError("readiness thresholds must be finite nonnegative numbers")
    for name in ("require_replay_comparison", "require_training_throughput", "require_puffer_export"):
        if type(getattr(args, name)) is not bool:
            raise ValueError(f"{name} must be a boolean")


if __name__ == "__main__":
    main()
