from __future__ import annotations

from typing import Any


def candidate_gap_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    sim = sim_failures(candidate)
    logs = {label: log_gap_summary(item) for label, item in candidate.get("live_logs", {}).items()}
    blockers = primary_blockers(sim, logs)
    return {
        "passed": bool(candidate.get("passed", False)),
        "primary_blockers": blockers,
        "sim_failures": sim,
        "live_log_failures": logs,
        "counts": {
            "sim": len(sim),
            "shadow": sum(len(item["shadow"]) for item in logs.values()),
            "command": sum(len(item["command"]) for item in logs.values()),
            "crash_replay": sum(len(item["crash_replay"]) for item in logs.values()),
            "source": sum(len(item["source"]) for item in logs.values()),
        },
    }


def sim_failures(candidate: dict[str, Any]) -> list[str]:
    failures = []
    for backend, item in candidate.get("sim", {}).items():
        gate = item.get("gate", {})
        failures.extend(f"{backend}:{failure}" for failure in gate.get("failures", []))
    return failures


def log_gap_summary(item: dict[str, Any]) -> dict[str, Any]:
    source = item.get("source_failure_evidence", {})
    source_metrics = source.get("source", {})
    return {
        "failed_source": bool(item.get("failed_source", False)),
        "shadow": gate_failures(item.get("shadow", {}).get("gate", {})),
        "command": gate_failures(item.get("command_gate", {})),
        "crash_replay": gate_failures(item.get("crash_replay", {}).get("gate", {})),
        "source": gate_failures(source),
        "precontact_horizontal_speed_max_m_s": source_metrics.get("precontact_horizontal_speed_max_m_s"),
        "horizontal_min_mm": source_metrics.get("horizontal_min_mm"),
        "tilt_max_abs_deg": source_metrics.get("tilt_max_abs_deg"),
    }


def gate_failures(gate: dict[str, Any]) -> list[str]:
    return list(gate.get("failures", []))


def primary_blockers(sim: list[str], logs: dict[str, dict[str, Any]]) -> list[str]:
    blockers = []
    if sim:
        blockers.append(f"sim:{', '.join(sim[:4])}")
    command_transform = labelled_failures(logs, "command", ("missing_commander_pitch_sign", "commander_pitch_sign_mismatch"))
    if command_transform:
        blockers.append(f"command_transform:{', '.join(command_transform[:4])}")
    source_drift = labelled_failures(logs, "source", ("source_precontact_drift",))
    if source_drift:
        blockers.append(f"source_precontact_drift:{', '.join(source_drift[:4])}")
    shadow_sign = labelled_contains(logs, "shadow", "_sign")
    if shadow_sign:
        blockers.append(f"shadow_sign:{', '.join(shadow_sign[:4])}")
    command_authority = labelled_contains(logs, "command", ("action_saturation", "roll_pitch_rate", "thrust"))
    if command_authority:
        blockers.append(f"command_authority:{', '.join(command_authority[:4])}")
    crash = labelled_all(logs, "crash_replay")
    if crash:
        blockers.append(f"crash_replay:{', '.join(crash[:4])}")
    return blockers or ["none"]


def labelled_failures(logs: dict[str, dict[str, Any]], key: str, names: tuple[str, ...]) -> list[str]:
    return [
        f"{label}:{failure}"
        for label, item in logs.items()
        for failure in item[key]
        if failure in names
    ]


def labelled_contains(logs: dict[str, dict[str, Any]], key: str, fragments: str | tuple[str, ...]) -> list[str]:
    needles = (fragments,) if isinstance(fragments, str) else fragments
    return [
        f"{label}:{failure}"
        for label, item in logs.items()
        for failure in item[key]
        if any(fragment in failure for fragment in needles)
    ]


def labelled_all(logs: dict[str, dict[str, Any]], key: str) -> list[str]:
    return [f"{label}:{failure}" for label, item in logs.items() for failure in item[key]]
