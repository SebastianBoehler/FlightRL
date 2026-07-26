from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flightrl.sixdof.transfer_selection import split_label_path
from flightrl.sixdof.transfer_test import LiveLogCase, TransferTestConfig, load_live_rows
from flightrl.sixdof.velocity_transfer import VelocityTransferConfig, score_velocity_transfer_policy
try:
    from scripts.build_puffer_policy_bundle_transfer_report import (
        live_case,
        load_policy,
        obstacle_failures,
        obstacle_report,
        velocity_failures,
    )
except ModuleNotFoundError:
    from build_puffer_policy_bundle_transfer_report import (
        live_case,
        load_policy,
        obstacle_failures,
        obstacle_report,
        velocity_failures,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a multi-seed Puffer transfer robustness matrix.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--obstacle-checkpoint", required=True)
    parser.add_argument("--velocity-checkpoint", required=True)
    parser.add_argument("--obstacle-mode", default="obstacle_hover")
    parser.add_argument("--velocity-mode", default="velocity_target")
    parser.add_argument("--obstacle-live-log", action="append", default=[], help="LABEL:CSV")
    parser.add_argument("--failed-obstacle-live-log", action="append", default=[], help="LABEL:CSV")
    parser.add_argument("--velocity-live-log", action="append", default=[], help="LABEL:CSV")
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--physics-profile")
    parser.add_argument("--sensor-profile")
    parser.add_argument("--domain-randomization")
    parser.add_argument("--disturbance-profile", default="raw_live_drift")
    parser.add_argument("--reset-profile", default="obstacle_hover_live")
    args = parser.parse_args()

    report = build_report(args)
    write_report(report, Path(args.output))
    print(f"puffer_transfer_robustness_matrix={args.output}")
    print(f"passed={report['passed']}")


def build_report(args) -> dict[str, Any]:
    obstacle_logs = [live_case(item, False) for item in args.obstacle_live_log] + [live_case(item, True) for item in args.failed_obstacle_live_log]
    velocity_logs = [split_label_path(item) for item in args.velocity_live_log]
    obstacle_policy = load_policy(args.obstacle_checkpoint, args.obstacle_mode)
    velocity_policy = load_policy(args.velocity_checkpoint, args.velocity_mode)
    velocity_config = VelocityTransferConfig()
    velocity = {label: score_velocity_transfer_policy(velocity_policy, load_live_rows(path), velocity_config) for label, path in velocity_logs}
    runs = [
        seed_run(args, seed, obstacle_policy, obstacle_logs, velocity, velocity_config)
        for seed in args.seed
    ]
    passed = all(run["passed"] for run in runs)
    return {
        "label": args.label,
        "passed": passed,
        "summary": summary(runs),
        "seeds": args.seed,
        "config": vars(args),
        "velocity_config": asdict(velocity_config),
        "runs": runs,
        "safety": "Offline robustness matrix only; passing this report does not approve live hardware deployment.",
    }


def seed_run(
    args,
    seed: int,
    obstacle_policy,
    obstacle_logs: list[LiveLogCase],
    velocity: dict[str, Any],
    velocity_config: VelocityTransferConfig,
) -> dict[str, Any]:
    config = TransferTestConfig(
        steps=args.steps,
        num_envs=args.num_envs,
        seed=seed,
        reset_profile=args.reset_profile,
        physics_profile=args.physics_profile,
        sensor_profile=args.sensor_profile,
        domain_randomization=args.domain_randomization,
        disturbance_profile=args.disturbance_profile,
    )
    obstacle = obstacle_report(obstacle_policy, args.obstacle_checkpoint, obstacle_logs, config)
    velocity_passed = all(item["gate"]["passed"] for item in velocity.values())
    passed = obstacle["passed"] and velocity_passed
    return {
        "seed": seed,
        "passed": passed,
        "config": asdict(config),
        "obstacle": obstacle,
        "velocity": velocity,
        "velocity_config": asdict(velocity_config),
        "failures": seed_failures(obstacle, velocity),
    }


def seed_failures(obstacle: dict[str, Any], velocity: dict[str, Any]) -> list[str]:
    failures = []
    obstacle_text = obstacle_failures(obstacle)
    velocity_text = velocity_failures(velocity)
    if obstacle_text != "none":
        failures.append(f"obstacle:{obstacle_text}")
    if velocity_text != "none":
        failures.append(f"velocity:{velocity_text}")
    return failures


def summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for run in runs if run["passed"])
    failures = {failure for run in runs for failure in run["failures"]}
    return {
        "passed": passed,
        "total": len(runs),
        "pass_rate": float(passed / len(runs)) if runs else 0.0,
        "failures": sorted(failures),
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Puffer Transfer Robustness Matrix",
        "",
        f"Passed: `{report['passed']}`",
        f"Pass rate: `{report['summary']['passed']}/{report['summary']['total']}`",
        "",
        "| seed | passed | failures |",
        "| ---: | ---: | --- |",
    ]
    for run in report["runs"]:
        lines.append(f"| {run['seed']} | `{run['passed']}` | {', '.join(run['failures']) or 'none'} |")
    lines.extend(["", "## Summary", "", f"- Failures: `{', '.join(report['summary']['failures']) or 'none'}`", "", report["safety"]])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
