from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from math import radians
from pathlib import Path
from typing import Any

import numpy as np
import torch

from flightrl.hardware.sixdof_live_replay import action_columns, live_env_from_telemetry, value
from flightrl.sixdof import SixDofCrazyflieEnv, teacher_actions
from flightrl.sixdof.puffer_policy import load_puffer_sixdof_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a Puffer six-DoF checkpoint over live Crazyflie telemetry.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shadow-output")
    parser.add_argument("--task", default="obstacle_avoidance")
    parser.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.50])
    parser.add_argument("--target-yaw-deg", type=float, default=0.0)
    args = parser.parse_args()

    rows = load_rows(args.input)
    policy = load_puffer_sixdof_policy(args.checkpoint)
    report, shadow_rows = evaluate_rows(policy, rows, args)
    write_json(report, args.output)
    Path(args.output).with_suffix(".md").write_text(render_markdown(report) + "\n")
    if args.shadow_output:
        write_csv(args.shadow_output, shadow_rows)
    print(f"puffer_live_shadow_report={args.output}")
    print(f"samples={report['samples']} l2_p95={report['groups']['all'].get('l2_p95')}")


def load_rows(path: str | Path) -> list[dict[str, float]]:
    parsed = []
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            parsed.append({key: parse_float(value) for key, value in row.items()})
    return parsed


def parse_float(raw: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def evaluate_rows(policy, rows: list[dict[str, float]], args) -> tuple[dict[str, Any], list[dict[str, float]]]:
    env = SixDofCrazyflieEnv(num_envs=1, seed=0, task=args.task)
    target = np.asarray(args.target, dtype=np.float32)
    target_yaw = radians(args.target_yaw_deg)
    pairs: list[tuple[np.ndarray, np.ndarray, dict[str, float]]] = []
    shadow_rows = []
    with torch.no_grad():
        for row in rows:
            live_env_from_telemetry(env, row, target=target, target_yaw=target_yaw)
            obs = env.observation().astype(np.float32)
            teacher = teacher_actions(env, task=args.task)[0]
            action = policy(torch.from_numpy(obs)).cpu().numpy()[0]
            pairs.append((action, teacher, row))
            shadow_rows.append({**row, **action_columns("puffer", action), **action_columns("teacher", teacher)})
    return build_report(args, policy, pairs), shadow_rows


def build_report(args, policy, pairs: list[tuple[np.ndarray, np.ndarray, dict[str, float]]]) -> dict[str, Any]:
    groups = {
        "all": pairs,
        "close_lt_18cm": [pair for pair in pairs if min_range(pair[2]) < 0.18],
        "close_lt_32cm": [pair for pair in pairs if min_range(pair[2]) < 0.32],
        "urgent_ttc_lt_35": [pair for pair in pairs if value(pair[2], "min_horizontal_ttc_s") < 0.35],
    }
    return {
        "checkpoint": args.checkpoint,
        "input": args.input,
        "task": args.task,
        "samples": len(pairs),
        "policy": asdict(policy.metadata),
        "groups": {name: group_metrics(group) for name, group in groups.items()},
        "safety": "Replay-only shadow report; no live hardware commands were produced by this checkpoint.",
    }


def group_metrics(group: list[tuple[np.ndarray, np.ndarray, dict[str, float]]]) -> dict[str, Any]:
    if not group:
        return {"samples": 0}
    actions = np.asarray([item[0] for item in group], dtype=np.float32)
    teachers = np.asarray([item[1] for item in group], dtype=np.float32)
    errors = actions - teachers
    l2 = np.linalg.norm(errors, axis=1)
    return {
        "samples": int(len(group)),
        "l2_mean": float(np.mean(l2)),
        "l2_p95": float(np.quantile(l2, 0.95)),
        "mae": float(np.mean(np.abs(errors))),
        "action_abs_mean": float(np.mean(np.abs(actions))),
        "action_abs_max": float(np.max(np.abs(actions))),
        "saturation_fraction": float(np.mean(np.abs(actions) > 0.95)),
        "sign_agreement": {
            "roll_rate": sign_agreement(actions[:, 1], teachers[:, 1]),
            "pitch_rate": sign_agreement(actions[:, 2], teachers[:, 2]),
        },
    }


def min_range(row: dict[str, float]) -> float:
    if "min_horizontal_range_m" in row:
        return value(row, "min_horizontal_range_m")
    values = [value(row, key) / 1000.0 for key in ("range.front", "range.back", "range.left", "range.right")]
    return min(4.0 if item >= 32.0 else item for item in values)


def sign_agreement(actual: np.ndarray, expected: np.ndarray) -> float:
    mask = np.abs(expected) > 1e-4
    if not np.any(mask):
        return 1.0
    return float(np.mean(np.sign(actual[mask]) == np.sign(expected[mask])))


def write_json(report: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, float]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["host_time_s"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Puffer Live Shadow Report", "", f"- Samples: `{report['samples']}`", ""]
    lines.append("| group | samples | l2 p95 | action max | roll sign | pitch sign |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, metrics in report["groups"].items():
        if not metrics.get("samples"):
            lines.append(f"| {name} | 0 | n/a | n/a | n/a | n/a |")
            continue
        signs = metrics["sign_agreement"]
        lines.append(
            f"| {name} | {metrics['samples']} | {metrics['l2_p95']:.4f} | "
            f"{metrics['action_abs_max']:.4f} | {signs['roll_rate']:.3f} | {signs['pitch_rate']:.3f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
