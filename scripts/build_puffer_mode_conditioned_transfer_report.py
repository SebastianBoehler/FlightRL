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
    parser = argparse.ArgumentParser(description="Evaluate one mode-conditioned Puffer checkpoint on obstacle and velocity transfer gates.")
    parser.add_argument("--candidate", action="append", required=True, help="LABEL:CHECKPOINT")
    parser.add_argument("--obstacle-live-log", action="append", default=[], help="LABEL:CSV")
    parser.add_argument("--failed-obstacle-live-log", action="append", default=[], help="LABEL:CSV")
    parser.add_argument("--velocity-live-log", action="append", default=[], help="LABEL:CSV")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--disturbance-profile", default="raw_live_drift")
    parser.add_argument("--reset-profile", default="obstacle_hover_live")
    args = parser.parse_args()

    obstacle_logs = [live_case(item, False) for item in args.obstacle_live_log] + [live_case(item, True) for item in args.failed_obstacle_live_log]
    velocity_logs = [split_label_path(item) for item in args.velocity_live_log]
    transfer_config = TransferTestConfig(
        task="obstacle_avoidance",
        steps=args.steps,
        num_envs=args.num_envs,
        reset_profile=args.reset_profile,
        disturbance_profile=args.disturbance_profile,
    )
    velocity_config = VelocityTransferConfig()
    candidates = {}
    for item in args.candidate:
        label, checkpoint = split_label_path(item)
        policy = load_puffer_sixdof_policy(checkpoint)
        obstacle_policy = ModeConditionedWrapper(policy, "obstacle_hover")
        velocity_policy = ModeConditionedWrapper(policy, "velocity_target")
        obstacle = obstacle_report(obstacle_policy, checkpoint, obstacle_logs, transfer_config)
        velocity = {
            log_label: score_velocity_transfer_policy(velocity_policy, load_live_rows(path), velocity_config)
            for log_label, path in velocity_logs
        }
        candidates[label] = {
            "checkpoint": checkpoint,
            "observation_dim": policy.metadata.observation_dim,
            "obstacle": obstacle,
            "velocity": velocity,
            "passed": obstacle["passed"] and all(item["gate"]["passed"] for item in velocity.values()),
        }
    report = {
        "passed": all(item["passed"] for item in candidates.values()),
        "config": {"transfer": asdict(transfer_config), "velocity": asdict(velocity_config)},
        "candidates": candidates,
        "safety": "Offline mode-conditioned transfer report only; not approved for live hardware.",
    }
    write_report(report, Path(args.output))
    print(f"mode_conditioned_transfer_report={args.output}")
    print(f"passed={report['passed']}")


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
    return all(item.get("gate", {}).get("passed", False) for item in report["sim"].values()) and all(
        item["shadow"]["gate"]["passed"] and item["command_gate"]["passed"] and item.get("crash_replay", {"gate": {"passed": True}})["gate"]["passed"]
        for item in report["live_logs"].values()
    )


def live_case(item: str, failed_source: bool) -> LiveLogCase:
    label, path = split_label_path(item)
    return LiveLogCase(label=label, path=path, failed_source=failed_source)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Puffer Mode-Conditioned Transfer Report", "", f"Passed: `{report['passed']}`", ""]
    lines.append("| candidate | passed | obstacle passed | velocity passed | obstacle failures | velocity failures |")
    lines.append("| --- | ---: | ---: | ---: | --- | --- |")
    for label, candidate in report["candidates"].items():
        velocity_passed = all(item["gate"]["passed"] for item in candidate["velocity"].values())
        lines.append(
            f"| {label} | `{candidate['passed']}` | `{candidate['obstacle']['passed']}` | `{velocity_passed}` | "
            f"{obstacle_failures(candidate['obstacle'])} | {velocity_failures(candidate['velocity'])} |"
        )
    lines.extend(["", "## Velocity Metrics", ""])
    lines.append("| candidate | log | horizontal l2 p95 | yaw abs p95 | vx sign | yaw sign |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for label, candidate in report["candidates"].items():
        for log_label, item in candidate["velocity"].items():
            policy = item["policy"]
            signs = policy.get("sign_agreement", {})
            lines.append(
                f"| {label} | {log_label} | {fmt(policy.get('horizontal_l2_p95_m_s'))} | "
                f"{fmt(policy.get('yaw_abs_p95_deg_s'))} | {fmt(signs.get('vx'))} | {fmt(signs.get('yawrate'))} |"
            )
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


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
