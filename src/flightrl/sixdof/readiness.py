from __future__ import annotations

import json
from pathlib import Path


def compact_residual_sweep(report: dict | None) -> dict:
    if not report:
        return {"present": False, "best": None}
    best = report.get("summary", {}).get("best")
    return {
        "present": True,
        "run": bool(report.get("run", False)),
        "total": report.get("summary", {}).get("total", len(report.get("records", []))),
        "completed": report.get("summary", {}).get("completed"),
        "thresholds": report.get("thresholds", {}),
        "best": compact_residual_best(best),
    }


def compact_residual_best(best: dict | None) -> dict | None:
    if not best:
        return None
    return {
        "name": best.get("name"),
        "checkpoint": best.get("checkpoint"),
        "passed": best.get("passed"),
        "mean_completed_fraction": best.get("mean_completed_fraction"),
        "mean_position_error_m": best.get("mean_position_error_m"),
        "mean_yaw_error_rad": best.get("mean_yaw_error_rad"),
        "teacher_action_l2_mean": best.get("teacher_action_l2_mean"),
    }


def compact_training_throughput(report: dict | None) -> dict:
    if not report:
        return {"present": False, "best_total_sps": None}
    summary = report.get("summary", {})
    best = summary.get("best_total_sps")
    return {
        "present": True,
        "controller": report.get("controller", "policy"),
        "residual_scale": report.get("residual_scale", 0.0),
        "tasks": report.get("tasks", []),
        "total": summary.get("total", len(report.get("records", []))),
        "best_total_sps": compact_throughput_best(best),
    }


def training_throughput_failures(report: dict, *, require: bool, min_total_sps: float) -> list[str]:
    if not require and min_total_sps <= 0.0:
        return []
    best = report.get("best_total_sps") or {}
    total_sps = best.get("total_sps")
    if not report.get("present") or total_sps is None:
        return ["training_throughput_missing"]
    if float(total_sps) < min_total_sps:
        return ["training_throughput_slow"]
    return []


def compact_puffer_export(report: dict | None) -> dict:
    if not report:
        return {"present": False, "passed": False}
    return {
        "present": True,
        "passed": bool(report.get("passed", False)),
        "env_name": report.get("env_name"),
        "checks": report.get("checks", []),
        "config": report.get("config", {}),
        "files": report.get("files", {}),
    }


def puffer_export_failures(report: dict, *, require: bool) -> list[str]:
    if not require:
        return []
    if not report.get("present"):
        return ["puffer_export_missing"]
    return [] if report.get("passed") else ["puffer_export"]


def summary(records: list[dict]) -> dict:
    ready = [record for record in records if record["ready"]]
    return {"total": len(records), "ready": len(ready), "blocked": len(records) - len(ready), "ready_tasks": [record["task"] for record in ready]}


def read_json(path: str | None) -> dict:
    return json.loads(Path(path).read_text()) if path else {}


def compact_throughput_best(best: dict | None) -> dict | None:
    if not best:
        return None
    return {
        "name": best.get("name"),
        "total_sps": best.get("total_sps"),
        "num_envs": best.get("num_envs"),
        "horizon": best.get("horizon"),
        "hidden_size": best.get("hidden_size"),
    }


def format_optional(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def format_task_gates(per_task: dict[str, dict]) -> str:
    return "; ".join(f"{task}={','.join(gate['failures']) or 'pass'}" for task, gate in per_task.items())


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
