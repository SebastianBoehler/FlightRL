from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize current FlightRL sim-to-real transfer gaps.")
    parser.add_argument("--readiness", required=True)
    parser.add_argument("--vertical-replay", required=True)
    parser.add_argument("--command-sweep", action="append", default=[])
    parser.add_argument("--profile-gate", action="append", default=[], help="Optional LABEL:JSON sim-gate report under a current sensor profile.")
    parser.add_argument("--output", default="artifacts/replay/sim2real_gap_diagnosis.json")
    args = parser.parse_args()

    report = build_report(
        readiness=read_json(args.readiness),
        vertical=read_json(args.vertical_replay),
        command_sweeps=[read_json(path) for path in args.command_sweep],
        profile_gates=[read_labelled_json(value) for value in args.profile_gate],
        paths=args,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"sim2real_gap_diagnosis={output}")
    print(f"markdown={output.with_suffix('.md')}")


def build_report(
    *,
    readiness: dict[str, Any],
    vertical: dict[str, Any],
    command_sweeps: list[dict[str, Any]],
    profile_gates: list[tuple[str, dict[str, Any]]],
    paths: argparse.Namespace,
) -> dict[str, Any]:
    sim_gate = readiness.get("sim_gate_summary", {})
    live_replay = readiness.get("heldout_live_replay_l2_p95", {})
    command = [summarize_command_sweep(report) for report in command_sweeps]
    ranked_profile_gates = summarize_profile_gates_ranked(profile_gates)
    findings = [
        "The current live replay P95 is policy-vs-reconstructed-six-DoF-teacher, not direct velocity-command reproduction.",
        "The aggressive live-replay fine-tune improves a local replay number but fails long-horizon Python/MuJoCo gates.",
        "The static box-room range model is not realistic for hand/cardboard obstacle logs; use range RMSE as a modeling warning, not as a pure controller score.",
        "State-bridge replay is much closer when z-hold is allowed, which supports keeping a hover-height prior in passive replay calibration.",
    ]
    return {
        "inputs": {
            "readiness": paths.readiness,
            "vertical_replay": paths.vertical_replay,
            "command_sweeps": paths.command_sweep,
        },
        "metric_definitions": {
            "heldout_live_replay_l2_p95": "Puffer policy action vs reconstructed six-DoF teacher action on logged telemetry.",
            "sim_gate_completion": "Closed-loop autonomous rollout completion in Python and MuJoCo backends.",
            "command_replay_state_bridge_score": "Passive command replay score without range RMSE: xy + 0.25*z + 0.01*yaw.",
            "command_replay_blended_score": "Passive command replay score including range RMSE from the current static room model.",
        },
        "vertical_clearance": {
            "summary": vertical.get("summary", {}),
            "model": vertical.get("model", {}),
        },
        "puffer_candidates": summarize_candidates(sim_gate, live_replay),
        "current_profile_gates": [summarize_profile_gate(label, gate) for label, gate in profile_gates],
        "command_replay": command,
        "diagnosis": {
            "status": readiness.get("status"),
            "findings": findings,
            "main_bottlenecks": [
                "Velocity-command live logs and six-DoF policy actions are different control spaces.",
                "Local imitation/replay matching can overfit and destabilize closed-loop simulation.",
                "Range observations in live obstacle tests include moving hands/cardboard, while the replay simulator currently assumes static geometry.",
                "Vertical top/bottom squeeze was a real controller-data flaw; the tuned model removes downward commands under floor guard.",
            ],
            "recommended_route": [
                "Keep native 6-DoF/Puffer as the main training spine.",
                "Use MuJoCo as a second closed-loop gate, not the primary optimizer.",
                "Use command replay state-bridge metrics to tune action/height/yaw bridge assumptions.",
                "Use direct live learned policy only after passing sim gates and live shadow on the same sensor profile.",
            ],
        },
        "next_live_readiness": {
            "ready_for_live_shadow": True,
            "ready_for_direct_learned_policy": False,
            "recommended_sequence": readiness.get("recommended_next_live_sequence", []),
            "recommended_shadow_comparison": [item["name"] for item in ranked_profile_gates],
            "primary_shadow_checkpoint": ranked_profile_gates[0]["checkpoint"] if ranked_profile_gates else readiness.get("recommended_shadow_checkpoint"),
            "legacy_shadow_checkpoint": readiness.get("recommended_shadow_checkpoint"),
        },
        "safety": "Offline diagnosis only; no live hardware commands were produced.",
    }


def summarize_candidates(sim_gate: dict[str, Any], live_replay: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for name, gate in sorted(sim_gate.items()):
        backends = gate.get("backends", {})
        records.append(
            {
                "name": name,
                "sim_passed": bool(gate.get("passed", False)),
                "heldout_l2_p95": live_replay.get(name) or live_replay.get(f"{name}_tuned_teacher"),
                "python": compact_backend(backends.get("python", {})),
                "mujoco": compact_backend(backends.get("mujoco", {})),
            }
        )
    return records


def compact_backend(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "completed_fraction": metrics.get("completed_fraction"),
        "position_error_m": metrics.get("position_error_m"),
        "clearance_p01_m": metrics.get("clearance_p01_m"),
        "survival_fraction": metrics.get("survival_fraction"),
        "action_saturation_fraction": metrics.get("action_saturation_fraction"),
        "failures": metrics.get("failures", []),
    }


def summarize_profile_gate(label: str, report: dict[str, Any]) -> dict[str, Any]:
    backends = report.get("reports", {})
    return {
        "name": label,
        "checkpoint": report.get("checkpoint"),
        "sensor_profile": report.get("sensor_profile"),
        "passed": bool(report.get("passed", False)),
        "python": compact_gate_backend(backends.get("python", {})),
        "mujoco": compact_gate_backend(backends.get("mujoco", {})),
    }


def summarize_profile_gates_ranked(profile_gates: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    records = [summarize_profile_gate(label, report) for label, report in profile_gates]
    return sorted(records, key=profile_gate_rank)


def profile_gate_rank(record: dict[str, Any]) -> tuple[int, float, float]:
    py = record["python"]
    return (
        0 if record["passed"] else 1,
        -(py.get("completed_fraction") or 0.0),
        py.get("position_error_m") or float("inf"),
    )


def compact_gate_backend(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    gate = report.get("gate", {})
    return {
        "passed": gate.get("passed"),
        "failures": gate.get("failures", []),
        "completed_fraction": metrics.get("mean_completed_fraction"),
        "position_error_m": metrics.get("mean_position_error_m"),
        "clearance_p01_m": metrics.get("clearance_p01_m"),
        "survival_fraction": metrics.get("mean_survival_fraction"),
        "action_saturation_fraction": metrics.get("action_saturation_fraction"),
        "teacher_action_l2_p95": metrics.get("teacher_action_l2_p95"),
    }


def summarize_command_sweep(report: dict[str, Any]) -> dict[str, Any]:
    records = report.get("records", [])
    return {
        "input": report.get("input"),
        "records": len(records),
        "best_blended": compact_command(best(records, lambda item: item.get("score", float("inf")))),
        "best_state_bridge": compact_command(best(records, lambda item: item["metrics"].get("state_bridge_score", float("inf")))),
        "best_xy_yaw": compact_command(best(records, lambda item: item["metrics"].get("xy_yaw_score", float("inf")))),
        "best_range": compact_command(best(records, lambda item: item["metrics"].get("worst_range_rmse_mm", float("inf")))),
    }


def compact_command(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not record:
        return None
    metrics = record.get("metrics", {})
    return {
        "params": record.get("params", {}),
        "score": record.get("score"),
        "state_bridge_score": metrics.get("state_bridge_score"),
        "xy_yaw_score": metrics.get("xy_yaw_score"),
        "xy_rmse_m": metrics.get("worst_xy_state_rmse_m"),
        "z_rmse_m": metrics.get("z_rmse_m"),
        "yaw_rmse_deg": metrics.get("yaw_rmse_deg"),
        "range_rmse_mm": metrics.get("worst_range_rmse_mm"),
        "overlap_duration_s": metrics.get("overlap_duration_s"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sim-to-Real Gap Diagnosis",
        "",
        f"- Status: `{report['diagnosis']['status']}`",
        f"- Live shadow ready: `{report['next_live_readiness']['ready_for_live_shadow']}`",
        f"- Direct learned policy ready: `{report['next_live_readiness']['ready_for_direct_learned_policy']}`",
        f"- Primary shadow checkpoint: `{report['next_live_readiness']['primary_shadow_checkpoint']}`",
        f"- Shadow comparison: `{', '.join(report['next_live_readiness']['recommended_shadow_comparison'])}`",
        "",
        "## Findings",
        "",
    ]
    lines.extend(f"- {item}" for item in report["diagnosis"]["findings"])
    lines.extend(["", "## Puffer/MuJoCo Candidates", ""])
    lines.extend(
        [
            "| candidate | sim | python completion | mujoco completion | heldout p95 | python pos err | mujoco pos err |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in report["puffer_candidates"]:
        lines.append(
            f"| {record['name']} | {record['sim_passed']} | {fmt(record['python']['completed_fraction'])} | "
            f"{fmt(record['mujoco']['completed_fraction'])} | {format_p95(record['heldout_l2_p95'])} | "
            f"{fmt(record['python']['position_error_m'])} | {fmt(record['mujoco']['position_error_m'])} |"
        )
    if report["current_profile_gates"]:
        lines.extend(["", "## Current Sensor Profile Gate", ""])
        lines.extend(
            [
                "| checkpoint | pass | python completion | mujoco completion | python pos err | mujoco pos err | python teacher p95 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for record in report["current_profile_gates"]:
            lines.append(
                f"| {record['name']} | {record['passed']} | {fmt(record['python']['completed_fraction'])} | "
                f"{fmt(record['mujoco']['completed_fraction'])} | {fmt(record['python']['position_error_m'])} | "
                f"{fmt(record['mujoco']['position_error_m'])} | {fmt(record['python']['teacher_action_l2_p95'])} |"
            )
    lines.extend(["", "## Command Replay", ""])
    lines.extend(
        [
            "| log | best state score | xy rmse | z rmse | yaw rmse | range rmse | params |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for sweep in report["command_replay"]:
        best_state = sweep.get("best_state_bridge") or {}
        params = best_state.get("params", {})
        lines.append(
            f"| {Path(sweep.get('input', '')).name} | {fmt(best_state.get('state_bridge_score'))} | "
            f"{fmt(best_state.get('xy_rmse_m'))} | {fmt(best_state.get('z_rmse_m'))} | "
            f"{fmt(best_state.get('yaw_rmse_deg'))} | {fmt(best_state.get('range_rmse_mm'))} | "
            f"{params.get('command_frame')}/{params.get('yaw_source')} hold_z={params.get('hold_z_m')} gain={params.get('velocity_gain')} sign={params.get('vx_sign')}/{params.get('vy_sign')} |"
        )
    lines.extend(["", "## Next Route", ""])
    lines.extend(f"- {item}" for item in report["diagnosis"]["recommended_route"])
    return "\n".join(lines)


def format_p95(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key}:{fmt(val)}" for key, val in sorted(value.items()))
    return fmt(value)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def best(records: list[dict[str, Any]], key) -> dict[str, Any] | None:
    return min(records, key=key) if records else None


def read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def read_labelled_json(value: str) -> tuple[str, dict[str, Any]]:
    label, sep, path = value.partition(":")
    if not sep:
        raise ValueError(f"expected LABEL:JSON for --profile-gate, got {value!r}")
    return label, read_json(path)


if __name__ == "__main__":
    main()
