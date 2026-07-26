from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flightrl.sixdof.action_targets import TARGET_SHAPINGS
from flightrl.sixdof.transfer_test import LiveLogCase, TransferTestConfig, evaluate_transfer_candidate
from flightrl.sixdof.transfer_log_quality import SourceTeacherQualityConfig, score_source_teacher_quality
from flightrl.sixdof.transfer_test import load_live_rows
from flightrl.sixdof.transfer_gap import candidate_gap_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Puffer sim-to-real transfer test report.")
    parser.add_argument("--candidate", action="append", required=True, help="LABEL:CHECKPOINT")
    parser.add_argument("--live-log", action="append", default=[], help="LABEL:CSV")
    parser.add_argument("--failed-live-log", action="append", default=[], help="LABEL:CSV")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="obstacle_avoidance")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--reset-profile", default="obstacle_hover_live")
    parser.add_argument("--physics-profile")
    parser.add_argument("--sensor-profile")
    parser.add_argument("--domain-randomization")
    parser.add_argument("--disturbance-profile")
    parser.add_argument("--max-open-space-horizontal-speed-p95-m-s", type=float, default=0.75)
    parser.add_argument("--target-mode", choices=("current_pose", "fixed_origin"), default="current_pose")
    parser.add_argument("--crash-target-shaping", choices=TARGET_SHAPINGS, default="none")
    parser.add_argument("--crash-target-shaping-strength", type=float, default=1.0)
    parser.add_argument("--previous-action-observation-scale", type=float, default=0.25)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    live_logs = [parse_live_case(item, False) for item in args.live_log] + [
        parse_live_case(item, True) for item in args.failed_live_log
    ]
    if not live_logs:
        raise SystemExit("at least one --live-log or --failed-live-log is required")
    config = TransferTestConfig(
        task=args.task,
        steps=args.steps,
        num_envs=args.num_envs,
        seed=args.seed,
        reset_profile=args.reset_profile,
        physics_profile=args.physics_profile,
        sensor_profile=args.sensor_profile,
        domain_randomization=args.domain_randomization,
        disturbance_profile=args.disturbance_profile,
        max_open_space_horizontal_speed_p95_m_s=args.max_open_space_horizontal_speed_p95_m_s,
        target_mode=args.target_mode,
        crash_target_shaping=args.crash_target_shaping,
        crash_target_shaping_strength=args.crash_target_shaping_strength,
        previous_action_observation_scale=args.previous_action_observation_scale,
    )
    candidates = {}
    for item in args.candidate:
        label, checkpoint = split_label_path(item, "--candidate")
        candidates[label] = evaluate_transfer_candidate(checkpoint, live_logs, config)
        candidates[label]["gap_summary"] = candidate_gap_summary(candidates[label])
    quality = source_teacher_quality(live_logs, config)
    report = {
        "passed": all(item["passed"] for item in candidates.values()) and source_quality_passed(quality),
        "config": vars(args),
        "live_logs": [asdict(case) for case in live_logs],
        "source_quality_passed": source_quality_passed(quality),
        "source_teacher_quality": quality,
        "candidates": candidates,
        "safety": "Offline transfer test only; passing this report does not approve live hardware deployment.",
    }
    write_report(report, Path(args.output))
    print(f"puffer_transfer_test={args.output}")
    print(f"passed={report['passed']}")
    if args.fail_on_gate and not report["passed"]:
        raise SystemExit(2)


def parse_live_case(item: str, failed_source: bool) -> LiveLogCase:
    label, path = split_label_path(item, "--live-log")
    return LiveLogCase(label=label, path=path, failed_source=failed_source)


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
    lines = ["# Puffer Transfer Test Set", "", f"Passed: `{report['passed']}`", ""]
    lines.append("| candidate | passed | sim failures | live shadow failures | command failures | crash replay failures |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for label, candidate in report["candidates"].items():
        lines.append(
            f"| {label} | `{candidate['passed']}` | {sim_failures(candidate)} | "
            f"{live_failures(candidate, 'shadow')} | {live_failures(candidate, 'command_gate')} | "
            f"{crash_replay_failures(candidate)} |"
        )
    lines.extend(["", "## Transfer Gap Summary", ""])
    lines.append("| candidate | blockers | sim | shadow | command | crash replay | source |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for label, candidate in report["candidates"].items():
        gap = candidate.get("gap_summary", {})
        counts = gap.get("counts", {})
        lines.append(
            f"| {label} | {', '.join(gap.get('primary_blockers', ['none']))} | "
            f"{counts.get('sim', 0)} | {counts.get('shadow', 0)} | {counts.get('command', 0)} | "
            f"{counts.get('crash_replay', 0)} | {counts.get('source', 0)} |"
        )
    lines.extend(["", "## Live Logs", ""])
    lines.append("| log | failed source | source failures | teacher/log l2 p95 | teacher/log thrust sign | roll sign | pitch sign | quality failures |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |")
    for case in report["live_logs"]:
        failures = source_failures(report, case["label"])
        quality = report.get("source_teacher_quality", {}).get(case["label"], {})
        signs = quality.get("sign_agreement", {})
        quality_failures = quality.get("gate", {}).get("failures", [])
        lines.append(
            f"| {case['label']} | `{case['failed_source']}` | {failures} | {fmt(quality.get('l2_p95'))} | "
            f"{fmt(signs.get('thrust'))} | {fmt(signs.get('roll_rate'))} | {fmt(signs.get('pitch_rate'))} | "
            f"{', '.join(quality_failures) or 'none'} |"
        )
    lines.extend(["", "## Candidate Metrics", ""])
    lines.append("| candidate | log | shadow scored/excluded | shadow l2 p95 | shadow action max | thrust sign | roll sign | pitch sign | command safe action p95 | crash l2 p95 | crash sat frac |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for candidate_label, candidate in report["candidates"].items():
        for log_label, item in candidate["live_logs"].items():
            shadow = item["shadow"]["groups"].get("all", {})
            command = item["command_gate"].get("safe", {})
            crash = item.get("crash_replay", {}).get("groups", {}).get("all", {})
            signs = shadow.get("sign_agreement", {})
            scored = item["shadow"].get("scored_samples", item["shadow"].get("samples", 0))
            excluded = item["shadow"].get("excluded_source_samples", 0)
            lines.append(
                f"| {candidate_label} | {log_label} | {scored}/{excluded} | {fmt(shadow.get('l2_p95'))} | "
                f"{fmt(shadow.get('action_abs_max'))} | {fmt(signs.get('thrust'))} | "
                f"{fmt(signs.get('roll_rate'))} | {fmt(signs.get('pitch_rate'))} | "
                f"{fmt(command.get('action_abs_p95'))} | "
                f"{fmt(crash.get('l2_p95'))} | {fmt(crash.get('saturation_fraction'))} |"
            )
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def sim_failures(candidate: dict[str, Any]) -> str:
    failures = []
    for backend, item in candidate["sim"].items():
        gate = item.get("gate", {})
        failures.extend(f"{backend}:{failure}" for failure in gate.get("failures", []))
    return ", ".join(failures) or "none"


def live_failures(candidate: dict[str, Any], key: str) -> str:
    failures = []
    for label, item in candidate["live_logs"].items():
        gate = item[key]["gate"] if key == "shadow" else item[key]
        failures.extend(f"{label}:{failure}" for failure in gate.get("failures", []))
    return ", ".join(failures) or "none"


def source_failures(report: dict[str, Any], label: str) -> str:
    failures = []
    for candidate in report["candidates"].values():
        item = candidate["live_logs"].get(label, {})
        source = item.get("source_failure_evidence", {})
        failures.extend(source.get("failures", []))
    return ", ".join(sorted(set(failures))) or "none"


def source_teacher_quality(live_logs: list[LiveLogCase], config: TransferTestConfig) -> dict[str, Any]:
    quality_config = SourceTeacherQualityConfig(task=config.task, target=config.target, target_yaw_deg=config.target_yaw_deg)
    return {
        case.label: {"samples": 0, "skipped": "failed_source", "gate": {"passed": True, "failures": []}}
        if case.failed_source
        else score_source_teacher_quality(load_live_rows(case.path), quality_config)
        for case in live_logs
    }


def source_quality_passed(quality: dict[str, Any]) -> bool:
    return all(item.get("gate", {}).get("passed", False) for item in quality.values())


def crash_replay_failures(candidate: dict[str, Any]) -> str:
    failures = []
    for label, item in candidate["live_logs"].items():
        gate = item.get("crash_replay", {}).get("gate", {})
        failures.extend(f"{label}:{failure}" for failure in gate.get("failures", []))
    return ", ".join(failures) or "none"


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    main()
