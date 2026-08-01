from __future__ import annotations

import json
from pathlib import Path

from flightrl.evidence_values import exact_nonnegative_int, exact_true, failure_strings, finite_number
from flightrl.sixdof.controller import CONTROLLERS
from flightrl.sixdof.tasks import TASKS


def compact_residual_sweep(report: dict | None) -> dict:
    if report is None:
        return {"present": False, "best": None}
    if not isinstance(report, dict):
        raise ValueError("residual sweep report must be an object")
    raw_summary = report.get("summary", {})
    report_summary = raw_summary if isinstance(raw_summary, dict) else {}
    best = report_summary.get("best")
    return {
        "present": True,
        "run": exact_true(report.get("run")),
        "total": report_summary.get("total", len(report.get("records", []))),
        "completed": report_summary.get("completed"),
        "thresholds": report.get("thresholds", {}),
        "best": compact_residual_best(best),
    }


def compact_residual_best(best: dict | None) -> dict | None:
    if not best:
        return None
    if not isinstance(best, dict):
        return None
    return {
        "name": best.get("name"),
        "checkpoint": best.get("checkpoint"),
        "passed": exact_true(best.get("passed")),
        "mean_completed_fraction": best.get("mean_completed_fraction"),
        "mean_position_error_m": best.get("mean_position_error_m"),
        "mean_yaw_error_rad": best.get("mean_yaw_error_rad"),
        "teacher_action_l2_mean": best.get("teacher_action_l2_mean"),
    }


def compact_training_throughput(report: dict | None) -> dict:
    if report is None:
        return {"present": False, "best_total_sps": None}
    if not isinstance(report, dict):
        raise ValueError("training throughput report must be an object")
    raw_summary = report.get("summary", {})
    summary = raw_summary if isinstance(raw_summary, dict) else {}
    best = summary.get("best_total_sps")
    compact_best = compact_throughput_best(best)
    records = report.get("records")
    total = exact_nonnegative_int(summary.get("total"))
    controller = report.get("controller")
    tasks = report.get("tasks")
    residual_scale = finite_number(report.get("residual_scale"))
    contract_valid = (
        controller in CONTROLLERS
        and isinstance(tasks, list)
        and bool(tasks)
        and all(isinstance(task, str) for task in tasks)
        and len(tasks) == len(set(tasks))
        and all(task in TASKS for task in tasks)
        and residual_scale is not None
        and residual_scale >= 0.0
    )
    valid = (
        ("run" not in report or exact_true(report.get("run")))
        and failure_strings(report.get("failures", [])) == []
        and failure_strings(summary.get("failures", [])) == []
        and isinstance(records, list)
        and total is not None
        and total > 0
        and total == len(records)
        and valid_throughput_best(compact_best)
        and contract_valid
    )
    return {
        "present": True,
        "valid": valid,
        "controller": controller,
        "residual_scale": residual_scale,
        "tasks": tasks if isinstance(tasks, list) else [],
        "total": total,
        "best_total_sps": compact_best,
    }


def training_throughput_failures(
    report: dict,
    *,
    require: bool,
    min_total_sps: float,
    controller: object = None,
    tasks: object = None,
) -> list[str]:
    minimum = finite_number(min_total_sps)
    if minimum is None or minimum < 0.0:
        raise ValueError("min_training_total_sps must be a finite nonnegative number")
    if type(require) is not bool:
        raise ValueError("require_training_throughput must be a boolean")
    if not require and minimum <= 0.0:
        return []
    if not isinstance(report, dict):
        return ["training_throughput_missing"]
    best = report.get("best_total_sps") or {}
    if not isinstance(best, dict):
        return ["training_throughput_missing"]
    total_sps = best.get("total_sps")
    measured = finite_number(total_sps)
    if not exact_true(report.get("present")) or not exact_true(report.get("valid")) or measured is None or measured <= 0.0:
        return ["training_throughput_missing"]
    if report.get("controller") != controller or report.get("tasks") != tasks:
        return ["training_throughput_contract"]
    if measured < minimum:
        return ["training_throughput_slow"]
    return []


def summary(records: list[dict]) -> dict:
    ready = [record for record in records if exact_true(record.get("ready"))]
    return {"total": len(records), "ready": len(ready), "blocked": len(records) - len(ready), "ready_tasks": [record["task"] for record in ready]}


def read_json(path: str | None) -> dict:
    data = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(data, dict):
        raise ValueError(f"evidence report must be a JSON object: {path}")
    return data


def compact_throughput_best(best: dict | None) -> dict | None:
    if not best:
        return None
    if not isinstance(best, dict):
        return None
    return {
        "name": best.get("name"),
        "total_sps": finite_number(best.get("total_sps")),
        "num_envs": best.get("num_envs"),
        "horizon": best.get("horizon"),
        "hidden_size": best.get("hidden_size"),
    }


def valid_throughput_best(best: dict | None) -> bool:
    if not isinstance(best, dict) or not isinstance(best.get("name"), str) or not best["name"]:
        return False
    if finite_number(best.get("total_sps")) is None or best["total_sps"] <= 0.0:
        return False
    return all(
        (value := exact_nonnegative_int(best.get(key))) is not None and value > 0
        for key in ("num_envs", "horizon", "hidden_size")
    )


def format_optional(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def format_task_gates(per_task: dict[str, dict]) -> str:
    return "; ".join(f"{task}={','.join(gate['failures']) or 'pass'}" for task, gate in per_task.items())


def render_markdown(report: dict) -> str:
    lines = [
        "# 6-DoF Desktop Development Readiness Report",
        "",
        "| scope | tasks | label | desktop ready | failures | desktop latency us | completed | pos err m | clearance p01 m |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for record in report["records"]:
        latency = record["desktop_latency"].get("per_sample_us")
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
