from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HARDWARE_MARKERS = ("require_cflib", "sync_crazyflie_context")
HARDWARE_TOUCH_MARKERS = ("require_cflib()", "sync_crazyflie_context(")
CHECKPOINT_MARKERS = ("--checkpoint", "torch.load", "create_policy_for_checkpoint")
CONTROL_MARKERS = ("send_", "motion_commander_cls", "start_linear_motion", "execute_demo_flight", "commander.")
APPROVAL_MARKERS = ("require_hardware_approved", "require_policy_approval")
MONITOR_MARKERS = ("hardware_approval_status", "monitor_only")
LEARNED_POLICY_MONITOR_MARKERS = ("LEARNED_POLICY_MONITOR_ONLY = True",)


def build_live_safety_report(paths: list[Path]) -> dict[str, Any]:
    records = [scan_live_script(path) for path in paths]
    failures = [failure for record in records for failure in record["failures"]]
    return {
        "summary": {
            "checked": len(records),
            "passed": not failures,
            "hardware_scripts": sum(1 for record in records if record["uses_hardware"]),
            "learned_checkpoint_hardware_scripts": sum(1 for record in records if record["uses_hardware"] and record["uses_checkpoint"]),
            "failures": failures,
        },
        "records": records,
        "safety": "Diagnostic source inventory only; typed edge-v3 bundle authority must not rely on lexical scanning.",
    }


def scan_live_script(path: Path) -> dict[str, Any]:
    text = path.read_text()
    uses_hardware = contains_any(text, HARDWARE_MARKERS)
    uses_checkpoint = contains_any(text, CHECKPOINT_MARKERS)
    controls_drone = contains_any(text, CONTROL_MARKERS)
    approval_gate = contains_any(text, APPROVAL_MARKERS)
    monitor_only = contains_any(text, LEARNED_POLICY_MONITOR_MARKERS) or (
        contains_any(text, MONITOR_MARKERS) and not controls_drone
    )
    failures = live_failures(path, text, uses_hardware, uses_checkpoint, controls_drone, approval_gate, monitor_only)
    return {
        "path": str(path),
        "uses_hardware": uses_hardware,
        "uses_checkpoint": uses_checkpoint,
        "controls_drone": controls_drone,
        "approval_gate": approval_gate,
        "monitor_only": monitor_only,
        "passed": not failures,
        "failures": failures,
    }


def live_failures(
    path: Path,
    text: str,
    uses_hardware: bool,
    uses_checkpoint: bool,
    controls_drone: bool,
    approval_gate: bool,
    monitor_only: bool,
) -> list[str]:
    if not uses_hardware or not uses_checkpoint:
        return []
    failures = []
    if controls_drone and not monitor_only:
        if not approval_gate:
            failures.append(f"{path}:checkpoint_control_without_hardware_approval")
        elif first_index(text, APPROVAL_MARKERS) > first_index(text, HARDWARE_TOUCH_MARKERS):
            failures.append(f"{path}:approval_after_cflib")
    elif not monitor_only:
        failures.append(f"{path}:checkpoint_monitor_without_monitor_only_metadata")
    return failures


def contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def first_index(text: str, markers: tuple[str, ...]) -> int:
    indexes = [text.find(marker) for marker in markers if marker in text]
    return min(indexes) if indexes else len(text) + 1


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Live Hardware Safety Report",
        "",
        f"- Passed: `{report['summary']['passed']}`",
        f"- Hardware scripts: `{report['summary']['hardware_scripts']}`",
        f"- Learned-checkpoint hardware scripts: `{report['summary']['learned_checkpoint_hardware_scripts']}`",
        "",
        "| script | hardware | checkpoint | control | approval gate | monitor only | passed | failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in report["records"]:
        lines.append(
            f"| `{record['path']}` | {record['uses_hardware']} | {record['uses_checkpoint']} | {record['controls_drone']} | "
            f"{record['approval_gate']} | {record['monitor_only']} | {record['passed']} | {', '.join(record['failures']) or 'none'} |"
        )
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
