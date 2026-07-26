from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flightrl.sixdof.action_targets import TARGET_SHAPINGS
from flightrl.sixdof.crash_replay import (
    CrashReplayConfig,
    build_crash_replay_dataset,
    score_crash_replay_policy,
    write_crash_replay_dataset,
)
from flightrl.sixdof.puffer_policy import load_puffer_sixdof_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and score a Puffer crash-state replay set from failed live logs.")
    parser.add_argument("--log", action="append", required=True, help="LABEL:CSV")
    parser.add_argument("--candidate", action="append", default=[], help="LABEL:CHECKPOINT")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-output")
    parser.add_argument("--task", default="obstacle_avoidance")
    parser.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.50])
    parser.add_argument("--target-yaw-deg", type=float, default=0.0)
    parser.add_argument("--target-shaping", choices=TARGET_SHAPINGS, default="none")
    parser.add_argument("--target-shaping-strength", type=float, default=1.0)
    parser.add_argument("--precontact-target-clip-abs", type=float, default=0.65)
    parser.add_argument("--close-target-clip-abs", type=float, default=0.85)
    parser.add_argument("--unsafe-target-clip-abs", type=float, default=0.45)
    parser.add_argument("--precontact-weight", type=float, default=2.0)
    parser.add_argument("--close-weight", type=float, default=1.0)
    parser.add_argument("--unsafe-weight", type=float, default=0.6)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    labelled_rows = [(label, path, load_rows(path)) for label, path in [split_label_path(item, "--log") for item in args.log]]
    rows = [row for _label, _path, items in labelled_rows for row in items]
    config = CrashReplayConfig(
        task=args.task,
        target=tuple(args.target),
        target_yaw_deg=args.target_yaw_deg,
        target_shaping=args.target_shaping,
        target_shaping_strength=args.target_shaping_strength,
        precontact_target_clip_abs=args.precontact_target_clip_abs,
        close_target_clip_abs=args.close_target_clip_abs,
        unsafe_target_clip_abs=args.unsafe_target_clip_abs,
        precontact_weight=args.precontact_weight,
        close_weight=args.close_weight,
        unsafe_weight=args.unsafe_weight,
    )
    dataset = build_crash_replay_dataset(rows, config)
    if args.dataset_output:
        write_crash_replay_dataset(args.dataset_output, dataset)
    candidates = {}
    for item in args.candidate:
        label, checkpoint = split_label_path(item, "--candidate")
        candidates[label] = score_crash_replay_policy(load_puffer_sixdof_policy(checkpoint), rows, config)
        candidates[label]["checkpoint"] = checkpoint
    report = {
        "passed": all(item["gate"]["passed"] for item in candidates.values()) if candidates else False,
        "config": asdict(config),
        "logs": [{"label": label, "path": path, "rows": len(items)} for label, path, items in labelled_rows],
        "dataset": dataset["summary"],
        "dataset_output": args.dataset_output,
        "candidates": candidates,
        "safety": "Offline crash-state replay only; passing this report does not approve live hardware deployment.",
    }
    write_report(report, Path(args.output))
    print(f"puffer_crash_replay={args.output}")
    print(f"samples={dataset['summary']['samples']} passed={report['passed']}")
    if args.fail_on_gate and not report["passed"]:
        raise SystemExit(2)


def load_rows(path: str | Path) -> list[dict[str, float]]:
    latest: dict[str, float] = {}
    parsed = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            latest.update({key: parse_float(raw) for key, raw in row.items() if raw != ""})
            parsed.append(dict(latest))
    return parsed


def parse_float(raw: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def split_label_path(item: str, flag: str) -> tuple[str, str]:
    if ":" not in item:
        raise SystemExit(f"{flag} must be LABEL:PATH")
    label, path = item.split(":", 1)
    if not label or not path:
        raise SystemExit(f"{flag} must be LABEL:PATH")
    return label, path


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Puffer Crash-State Replay Set", "", f"Passed: `{report['passed']}`", ""]
    lines.append("| log | rows |")
    lines.append("| --- | ---: |")
    for item in report["logs"]:
        lines.append(f"| {item['label']} | {item['rows']} |")
    counts = report["dataset"]["group_counts"]
    lines.extend(
        [
            "",
            "## Dataset",
            "",
            f"- Samples: `{report['dataset']['samples']}`",
            f"- Pre-contact drift: `{counts.get('precontact_drift', 0)}`",
            f"- Close recovery: `{counts.get('close_recovery', 0)}`",
            f"- Unsafe tail: `{counts.get('unsafe_tail', 0)}`",
            "",
            "## Candidates",
            "",
            "| candidate | passed | samples | all l2 p95 | precontact l2 p95 | action p95/max | sat frac | failures |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for label, candidate in report["candidates"].items():
        all_group = candidate["groups"].get("all", {})
        precontact = candidate["groups"].get("precontact_drift", {})
        lines.append(
            f"| {label} | `{candidate['gate']['passed']}` | {candidate['samples']} | "
            f"{fmt(all_group.get('l2_p95'))} | {fmt(precontact.get('l2_p95'))} | "
            f"{fmt(all_group.get('action_abs_p95'))}/{fmt(all_group.get('action_abs_max'))} | "
            f"{fmt(all_group.get('saturation_fraction'))} | {', '.join(candidate['gate']['failures']) or 'none'} |"
        )
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    main()
