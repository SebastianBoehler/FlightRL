from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flightrl.hardware.direct_raw_gate import DirectRawGateThresholds, evaluate_direct_raw_replay
from flightrl.sixdof.crash_replay import score_crash_replay_policy
from flightrl.sixdof.mode_conditioned import ModeConditionedWrapper
from flightrl.sixdof.puffer_evaluation import PufferEvalConfig, evaluate_puffer_backends
from flightrl.sixdof.puffer_policy import load_puffer_sixdof_policy
from flightrl.sixdof.transfer_selection import split_label_path
from flightrl.sixdof.transfer_test import (
    LiveLogCase,
    TransferTestConfig,
    crash_config_from_transfer,
    live_shadow_report,
    load_live_rows,
    raw_shadow_rows,
)
from flightrl.sixdof.velocity_transfer import VelocityTransferConfig, score_velocity_transfer_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an offline Puffer transfer bundle with separate obstacle and velocity policies.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--obstacle-checkpoint", required=True)
    parser.add_argument("--velocity-checkpoint", required=True)
    parser.add_argument("--obstacle-mode", default="obstacle_hover")
    parser.add_argument("--velocity-mode", default="velocity_target")
    parser.add_argument("--obstacle-live-log", action="append", default=[], help="LABEL:CSV")
    parser.add_argument("--failed-obstacle-live-log", action="append", default=[], help="LABEL:CSV")
    parser.add_argument("--velocity-live-log", action="append", default=[], help="LABEL:CSV")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--physics-profile")
    parser.add_argument("--sensor-profile")
    parser.add_argument("--domain-randomization")
    parser.add_argument("--disturbance-profile", default="raw_live_drift")
    parser.add_argument("--reset-profile", default="obstacle_hover_live")
    args = parser.parse_args()

    transfer_config = TransferTestConfig(
        task="obstacle_avoidance",
        steps=args.steps,
        num_envs=args.num_envs,
        reset_profile=args.reset_profile,
        physics_profile=args.physics_profile,
        sensor_profile=args.sensor_profile,
        domain_randomization=args.domain_randomization,
        disturbance_profile=args.disturbance_profile,
    )
    velocity_config = VelocityTransferConfig()
    obstacle_logs = [live_case(item, False) for item in args.obstacle_live_log] + [live_case(item, True) for item in args.failed_obstacle_live_log]
    velocity_logs = [split_label_path(item) for item in args.velocity_live_log]
    obstacle_policy = load_policy(args.obstacle_checkpoint, args.obstacle_mode)
    velocity_policy = load_policy(args.velocity_checkpoint, args.velocity_mode)
    obstacle = obstacle_report(obstacle_policy, args.obstacle_checkpoint, obstacle_logs, transfer_config)
    velocity = {label: score_velocity_transfer_policy(velocity_policy, load_live_rows(path), velocity_config) for label, path in velocity_logs}
    bundle = {
        "label": args.label,
        "obstacle_checkpoint": args.obstacle_checkpoint,
        "velocity_checkpoint": args.velocity_checkpoint,
        "obstacle": obstacle,
        "velocity": velocity,
        "passed": obstacle["passed"] and all(item["gate"]["passed"] for item in velocity.values()),
    }
    report = {
        "passed": bundle["passed"],
        "config": {"transfer": asdict(transfer_config), "velocity": asdict(velocity_config)},
        "bundle": bundle,
        "safety": "Offline bundle transfer report only; passing this report does not approve live hardware deployment.",
    }
    write_report(report, Path(args.output))
    print(f"puffer_policy_bundle_transfer_report={args.output}")
    print(f"passed={report['passed']}")


def load_policy(checkpoint: str, mode: str):
    policy = load_puffer_sixdof_policy(checkpoint)
    if policy.metadata.observation_dim == 28:
        return policy
    if policy.metadata.observation_dim == 30:
        return ModeConditionedWrapper(policy, mode)
    raise SystemExit(f"unsupported Puffer observation dim {policy.metadata.observation_dim} for {checkpoint}")


def obstacle_report(policy, checkpoint: str, logs: list[LiveLogCase], config: TransferTestConfig) -> dict[str, Any]:
    report = {
        "checkpoint": checkpoint,
        "sim": evaluate_puffer_backends(
            policy,
            PufferEvalConfig(
                task=config.task,
                backend="both",
                steps=config.steps,
                num_envs=config.num_envs,
                reset_profile=config.reset_profile,
                physics_profile=config.physics_profile,
                sensor_profile=config.sensor_profile,
                domain_randomization=config.domain_randomization,
                disturbance_profile=config.disturbance_profile,
                max_open_space_horizontal_speed_p95_m_s=config.max_open_space_horizontal_speed_p95_m_s,
                previous_action_observation_scale=config.previous_action_observation_scale,
            ),
        ),
        "live_logs": {},
    }
    for case in logs:
        rows = load_live_rows(case.path)
        item = {
            "path": case.path,
            "failed_source": case.failed_source,
            "shadow": live_shadow_report(policy, rows, case, config),
            "command_gate": evaluate_direct_raw_replay(
                raw_shadow_rows(policy, rows, config),
                DirectRawGateThresholds(min_safe_rows=config.min_command_safe_rows, require_source_health=False),
            ),
        }
        if case.failed_source:
            item["source_failure_evidence"] = evaluate_direct_raw_replay(
                rows,
                DirectRawGateThresholds(min_safe_rows=0, require_commander_pitch_sign=False),
            )
            item["crash_replay"] = score_crash_replay_policy(
                policy,
                rows,
                crash_config_from_transfer(config),
                previous_action_observation_scale=config.previous_action_observation_scale,
            )
        report["live_logs"][case.label] = item
    report["passed"] = obstacle_passed(report)
    return report


def obstacle_passed(report: dict[str, Any]) -> bool:
    sim_passed = all(item.get("gate", {}).get("passed", False) for item in report["sim"].values())
    live_passed = all(
        item["shadow"]["gate"]["passed"]
        and item["command_gate"]["passed"]
        and item.get("crash_replay", {"gate": {"passed": True}})["gate"]["passed"]
        for item in report["live_logs"].values()
    )
    return bool(sim_passed and live_passed)


def live_case(item: str, failed_source: bool) -> LiveLogCase:
    label, path = split_label_path(item)
    return LiveLogCase(label=label, path=path, failed_source=failed_source)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")


def render_markdown(report: dict[str, Any]) -> str:
    bundle = report["bundle"]
    velocity_passed = all(item["gate"]["passed"] for item in bundle["velocity"].values())
    lines = [
        "# Puffer Policy Bundle Transfer Report",
        "",
        f"Passed: `{report['passed']}`",
        "",
        "| bundle | passed | obstacle passed | velocity passed | obstacle checkpoint | velocity checkpoint |",
        "| --- | ---: | ---: | ---: | --- | --- |",
        f"| {bundle['label']} | `{bundle['passed']}` | `{bundle['obstacle']['passed']}` | `{velocity_passed}` | "
        f"`{bundle['obstacle_checkpoint']}` | `{bundle['velocity_checkpoint']}` |",
        "",
        "## Failures",
        "",
        f"- Obstacle: {obstacle_failures(bundle['obstacle'])}",
        f"- Velocity: {velocity_failures(bundle['velocity'])}",
        "",
    ]
    lines.extend(obstacle_metric_lines(bundle["obstacle"]))
    lines.extend(
        [
            "",
            "## Velocity Metrics",
            "",
        ]
    )
    lines.extend(
        [
        "| log | horizontal l2 p95 | yaw abs p95 | vx sign | yaw sign |",
        "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, item in bundle["velocity"].items():
        policy = item["policy"]
        signs = policy.get("sign_agreement", {})
        lines.append(
            f"| {label} | {fmt(policy.get('horizontal_l2_p95_m_s'))} | {fmt(policy.get('yaw_abs_p95_deg_s'))} | "
            f"{fmt(signs.get('vx'))} | {fmt(signs.get('yawrate'))} |"
        )
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def obstacle_metric_lines(report: dict[str, Any]) -> list[str]:
    lines = [
        "## Obstacle Metrics",
        "",
        "| log | failed source | source failures | precontact speed max | shadow scored/excluded | shadow l2 p95 | command action p95 | crash l2 p95 | crash precontact l2 p95 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, item in report["live_logs"].items():
        shadow = item["shadow"]["groups"].get("all", {})
        command = item["command_gate"].get("safe", {})
        crash_groups = item.get("crash_replay", {}).get("groups", {})
        source = item.get("source_failure_evidence", {})
        scored = item["shadow"].get("scored_samples", item["shadow"].get("samples", 0))
        excluded = item["shadow"].get("excluded_source_samples", 0)
        lines.append(
            f"| {label} | `{item['failed_source']}` | {source_failures(source)} | "
            f"{fmt(source.get('source', {}).get('precontact_horizontal_speed_max_m_s'))} | {scored}/{excluded} | "
            f"{fmt(shadow.get('l2_p95'))} | {fmt(command.get('action_abs_p95'))} | "
            f"{fmt(crash_groups.get('all', {}).get('l2_p95'))} | "
            f"{fmt(crash_groups.get('precontact_drift', {}).get('l2_p95'))} |"
        )
    return lines


def source_failures(source: dict[str, Any]) -> str:
    return ", ".join(source.get("failures", [])) or "none"


def obstacle_failures(report: dict[str, Any]) -> str:
    failures = []
    for backend, item in report["sim"].items():
        failures.extend(f"{backend}:{failure}" for failure in item.get("gate", {}).get("failures", []))
    for label, item in report["live_logs"].items():
        failures.extend(f"{label}:shadow:{failure}" for failure in item["shadow"]["gate"].get("failures", []))
        failures.extend(f"{label}:command:{failure}" for failure in item["command_gate"].get("failures", []))
        failures.extend(f"{label}:crash:{failure}" for failure in item.get("crash_replay", {}).get("gate", {}).get("failures", []))
    return ", ".join(failures) or "none"


def velocity_failures(reports: dict[str, Any]) -> str:
    failures = [f"{label}:{failure}" for label, item in reports.items() for failure in item["gate"]["failures"]]
    return ", ".join(failures) or "none"


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    main()
