from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flightrl.sixdof import checkpoint_tasks, load_controller_from_checkpoint
from flightrl.sixdof.diagnostics import diagnose_controller, summarize_diagnostics
from flightrl.sixdof.tasks import parse_task_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose 6-DoF policy failures by reset profile and rollout phase")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--task", required=True)
    parser.add_argument("--profiles", nargs="+", default=["position_yaw_easy", "position_yaw_medium", "position_yaw_wide", "broad"])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=700)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--native-step", action="store_true")
    parser.add_argument("--output", default="artifacts/replay/sixdof_policy_diagnostics.json")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu") if args.checkpoint else None
    if not args.teacher and checkpoint is None:
        raise SystemExit("--checkpoint is required unless --teacher is set")
    policy_tasks = parse_task_spec(args.task) if args.teacher else checkpoint_tasks(checkpoint)
    model = None if args.teacher else load_controller_from_checkpoint(checkpoint)
    if args.task not in policy_tasks:
        raise SystemExit(f"task {args.task!r} is not present in controller tasks {policy_tasks}")
    records = [
        diagnose_controller(
            model,
            policy_tasks,
            task=args.task,
            reset_profile=profile,
            seed=args.seed + index,
            steps=args.steps,
            num_envs=args.num_envs,
            observation_mode=(checkpoint or {}).get("observation_mode", "base"),
            use_native_step=args.native_step,
            bins=args.bins,
        )
        for index, profile in enumerate(args.profiles)
    ]
    report = {
        "checkpoint": args.checkpoint,
        "controller": "teacher" if args.teacher else checkpoint.get("controller", "checkpoint"),
        "task": args.task,
        "profiles": args.profiles,
        "summary": summarize_diagnostics(records),
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"diagnostics={output}")
    print(f"markdown={output.with_suffix('.md')}")


def render_markdown(report: dict) -> str:
    lines = [
        "# 6-DoF Policy Diagnostics",
        "",
        f"Controller: `{report['controller']}`",
        f"Task: `{report['task']}`",
        "",
        "| profile | survival | pos err mean m | pos err p95 m | clearance p01 m | yaw mean rad | yaw p95 rad | settled yaw p95 | action sat |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in report["records"]:
        final = record["final"]
        lines.append(
            f"| {record['reset_profile']} | {final['survival_fraction']:.4f} | {final['position_error_mean_m']:.4f} | "
            f"{final['position_error_p95_m']:.4f} | {final['clearance_p01_m']:.4f} | "
            f"{final['yaw_error_mean_rad']:.4f} | {final['yaw_error_p95_rad']:.4f} | "
            f"{record.get('phase_summary', {}).get('settled_half', {}).get('yaw_error_p95_rad', 0.0):.4f} | "
            f"{final['action_saturation_fraction']:.4f} |"
        )
    if report["summary"]["blocked"]:
        lines.extend(["", "## Blockers", ""])
        for item in report["summary"]["blocked"]:
            lines.append(f"- `{item['profile']}`: `{item['reason']}`")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
