from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flightrl.sixdof.puffer_policy import load_puffer_sixdof_policy
from flightrl.sixdof.transfer_selection import split_label_path
from flightrl.sixdof.transfer_test import load_live_rows
from flightrl.sixdof.velocity_transfer import VelocityTransferConfig, score_velocity_transfer_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Puffer velocity-command transfer gate from live velocity-control logs.")
    parser.add_argument("--candidate", action="append", required=True, help="LABEL:CHECKPOINT")
    parser.add_argument("--live-log", action="append", required=True, help="LABEL:CSV")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="obstacle_avoidance")
    parser.add_argument("--target-height-m", type=float, default=0.50)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--max-horizontal-l2-p95-m-s", type=float, default=0.08)
    parser.add_argument("--max-velocity-l2-p95-m-s", type=float, default=0.09)
    parser.add_argument("--max-yaw-abs-p95-deg-s", type=float, default=6.0)
    parser.add_argument("--min-vx-sign-agreement", type=float, default=0.55)
    parser.add_argument("--min-vy-sign-agreement", type=float, default=0.55)
    parser.add_argument("--min-yaw-sign-agreement", type=float, default=0.35)
    parser.add_argument("--max-horizontal-speed-m-s", type=float, default=0.12)
    parser.add_argument("--max-vertical-speed-m-s", type=float, default=0.04)
    parser.add_argument("--max-yawrate-deg-s", type=float, default=12.0)
    parser.add_argument("--rate-horizon-s", type=float, default=0.08)
    parser.add_argument("--max-virtual-tilt-rad", type=float, default=0.18)
    parser.add_argument("--horizontal-gain-s", type=float, default=0.06)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    config = config_from_args(args)
    logs = {label: load_live_rows(path) for label, path in [split_label_path(item) for item in args.live_log]}
    candidates = {}
    for item in args.candidate:
        label, checkpoint = split_label_path(item)
        policy = load_puffer_sixdof_policy(checkpoint)
        candidates[label] = {
            "checkpoint": checkpoint,
            "logs": {log_label: score_velocity_transfer_policy(policy, rows, config) for log_label, rows in logs.items()},
        }
        candidates[label]["passed"] = all(report["gate"]["passed"] for report in candidates[label]["logs"].values())
    report = {
        "passed": all(candidate["passed"] for candidate in candidates.values()),
        "config": asdict(config),
        "live_logs": [{"label": label, "rows": len(rows)} for label, rows in logs.items()],
        "candidates": candidates,
        "safety": "Offline velocity-command transfer gate only; passing this report does not approve live hardware deployment.",
    }
    write_report(report, Path(args.output))
    print(f"puffer_velocity_transfer_gate={args.output}")
    print(f"passed={report['passed']}")
    if args.fail_on_gate and not report["passed"]:
        raise SystemExit(2)


def config_from_args(args: argparse.Namespace) -> VelocityTransferConfig:
    return VelocityTransferConfig(
        task=args.task,
        target_height_m=args.target_height_m,
        min_samples=args.min_samples,
        max_horizontal_l2_p95_m_s=args.max_horizontal_l2_p95_m_s,
        max_velocity_l2_p95_m_s=args.max_velocity_l2_p95_m_s,
        max_yaw_abs_p95_deg_s=args.max_yaw_abs_p95_deg_s,
        min_vx_sign_agreement=args.min_vx_sign_agreement,
        min_vy_sign_agreement=args.min_vy_sign_agreement,
        min_yaw_sign_agreement=args.min_yaw_sign_agreement,
        max_horizontal_speed_m_s=args.max_horizontal_speed_m_s,
        max_vertical_speed_m_s=args.max_vertical_speed_m_s,
        max_yawrate_deg_s=args.max_yawrate_deg_s,
        rate_horizon_s=args.rate_horizon_s,
        max_virtual_tilt_rad=args.max_virtual_tilt_rad,
        horizontal_gain_s=args.horizontal_gain_s,
    )


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Puffer Velocity Transfer Gate", "", f"Passed: `{report['passed']}`", ""]
    lines.append("| candidate | log | passed | failures | horizontal l2 p95 | velocity l2 p95 | yaw abs p95 | vx sign | vy sign | yaw sign | source horizontal l2 p95 | source yaw abs p95 |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for candidate_label, candidate in report["candidates"].items():
        for log_label, item in candidate["logs"].items():
            policy = item["policy"]
            source = item["source_adapter"]
            signs = policy.get("sign_agreement", {})
            lines.append(
                f"| {candidate_label} | {log_label} | `{item['gate']['passed']}` | {', '.join(item['gate']['failures']) or 'none'} | "
                f"{fmt(policy.get('horizontal_l2_p95_m_s'))} | {fmt(policy.get('velocity_l2_p95_m_s'))} | "
                f"{fmt(policy.get('yaw_abs_p95_deg_s'))} | {fmt(signs.get('vx'))} | {fmt(signs.get('vy'))} | "
                f"{fmt(signs.get('yawrate'))} | {fmt(source.get('horizontal_l2_p95_m_s'))} | {fmt(source.get('yaw_abs_p95_deg_s'))} |"
            )
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    main()
