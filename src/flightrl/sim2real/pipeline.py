from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flightrl.sim2real.audit import build_audit, render_markdown as render_audit_markdown
from flightrl.sim2real.checkpoint_manifest import build_checkpoint_manifest, write_report as write_manifest_report
from flightrl.sim2real.data_plan import build_data_plan, render_markdown as render_data_plan_markdown
from flightrl.sim2real.live_safety import build_live_safety_report, write_report as write_live_safety_report
from flightrl.sim2real.profile import build_profile, write_report as write_profile_report
from flightrl.sim2real.profile_export import export_config, write_report as write_export_report
from flightrl.sim2real.provenance import path_provenance
from flightrl.sim2real.transfer_gate import build_transfer_gate, write_report as write_gate_report


def build_pipeline(
    *,
    outputs: dict[str, Path],
    hardware_config: Path,
    base_config: Path,
    output_config: Path,
    deployment_readiness: Path,
    sim_readiness: Path,
    live_scripts: list[Path],
    motor_calibration: Path | None = None,
    stationary_noise: Path | None = None,
    hardware_latency: Path | None = None,
    calibration_quality: Path | None = None,
    replay_comparison: Path | None = None,
    motor_bench: Path | None = None,
    room_report: Path | None = None,
    hardware_blockers: list[str] | None = None,
    input_paths: dict[str, Any] | None = None,
) -> dict[str, Any]:
    live_safety = build_live_safety_report(live_scripts)
    write_live_safety_report(live_safety, outputs["live_safety"])

    profile = build_profile(hardware_config=hardware_config, motor_calibration=motor_calibration, stationary_noise=stationary_noise, hardware_latency=hardware_latency)
    write_profile_report(profile, outputs["profile"])

    config_export = export_config(outputs["profile"], base_config=base_config, output_config=output_config)
    write_export_report(config_export, outputs["config_export"])

    audit = build_audit(
        hardware_config=hardware_config,
        calibration_quality=calibration_quality,
        deployment_readiness=deployment_readiness,
        replay_comparison=replay_comparison,
        motor_bench=motor_bench,
        stationary_noise=stationary_noise,
        hardware_latency=hardware_latency,
        hardware_blockers=hardware_blockers or [],
    )
    write_json_markdown(audit, outputs["audit"], render_audit_markdown)

    data_plan = build_data_plan(outputs["audit"], motor_bench=motor_bench)
    write_json_markdown(data_plan, outputs["data_plan"], render_data_plan_markdown)

    gate = build_transfer_gate(
        audit=outputs["audit"],
        profile=outputs["profile"],
        config_export=outputs["config_export"],
        deployment_readiness=deployment_readiness,
        sim_readiness=sim_readiness,
        room_report=room_report,
        live_safety=outputs["live_safety"],
    )
    write_gate_report(gate, outputs["transfer_gate"])

    manifest = build_checkpoint_manifest(transfer_gate=outputs["transfer_gate"], sim_readiness=sim_readiness, deployment_readiness=deployment_readiness)
    write_manifest_report(manifest, outputs["checkpoint_manifest"])

    inputs = input_paths or default_inputs(
        hardware_config=hardware_config,
        base_config=base_config,
        output_config=output_config,
        deployment_readiness=deployment_readiness,
        sim_readiness=sim_readiness,
        live_scripts=live_scripts,
        motor_calibration=motor_calibration,
        stationary_noise=stationary_noise,
        hardware_latency=hardware_latency,
        calibration_quality=calibration_quality,
        replay_comparison=replay_comparison,
        motor_bench=motor_bench,
        room_report=room_report,
        hardware_blockers=hardware_blockers or [],
    )
    report = pipeline_summary(outputs, audit, profile, config_export, gate, manifest, inputs)
    write_json_markdown(report, outputs["pipeline"], render_pipeline_markdown)
    return report


def pipeline_summary(
    outputs: dict[str, Path],
    audit: dict[str, Any],
    profile: dict[str, Any],
    config_export: dict[str, Any],
    gate: dict[str, Any],
    manifest: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "inputs": input_provenance(inputs),
        "artifacts": {key: str(path) for key, path in outputs.items()},
        "transfer_ready": bool(audit.get("transfer_ready", False)),
        "profile_ready": bool(profile.get("summary", {}).get("profile_ready", False)),
        "config_exported": bool(config_export.get("exported", False)),
        "transfer_approved": bool(gate.get("transfer_approved", False)),
        "hardware_approved_checkpoints": int(manifest.get("summary", {}).get("hardware_approved", 0) or 0),
        "blocking_items": audit.get("blocking_items", []),
        "gate_failures": gate.get("summary", {}).get("failures", []),
        "safety": "Pipeline rebuild is offline evidence generation only; it does not run live hardware.",
    }


def default_inputs(**kwargs) -> dict[str, Any]:
    return kwargs


def input_provenance(inputs: dict[str, Any]) -> dict[str, Any]:
    return {key: provenance_value(value) for key, value in inputs.items()}


def provenance_value(value: Any) -> Any:
    if isinstance(value, Path):
        return path_provenance(value)
    if isinstance(value, list):
        return [provenance_value(item) for item in value]
    if value is None:
        return {"path": None, "exists": False}
    return value


def output_paths(base_dir: Path, label: str) -> dict[str, Path]:
    return {
        "live_safety": base_dir / f"live_hardware_safety_{label}.json",
        "profile": base_dir / f"sim2real_profile_{label}.json",
        "config_export": base_dir / f"sim2real_config_export_{label}.json",
        "audit": base_dir / f"sim2real_audit_{label}.json",
        "data_plan": base_dir / f"sim2real_data_plan_{label}.json",
        "transfer_gate": base_dir / f"sim2real_transfer_gate_{label}.json",
        "checkpoint_manifest": base_dir / f"sim2real_checkpoint_manifest_{label}.json",
        "evidence_gap": base_dir / f"sim2real_evidence_gap_{label}.json",
        "pipeline": base_dir / f"sim2real_pipeline_{label}.json",
    }


def write_json_markdown(report: dict[str, Any], output: Path, renderer) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(renderer(report) + "\n")


def render_pipeline_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sim-To-Real Pipeline",
        "",
        f"- Transfer approved: `{report['transfer_approved']}`",
        f"- Hardware-approved checkpoints: `{report['hardware_approved_checkpoints']}`",
        f"- Profile ready: `{report['profile_ready']}`",
        f"- Config exported: `{report['config_exported']}`",
        "",
        "## Blocking Items",
        "",
    ]
    lines.extend(f"- `{item}`" for item in report["blocking_items"] or ["none"])
    lines.extend(["", "## Inputs", ""])
    for name, value in report["inputs"].items():
        lines.append(f"- `{name}`: {format_input(value)}")
    lines.extend(["", "## Artifacts", ""])
    lines.extend(f"- `{name}`: `{path}`" for name, path in report["artifacts"].items())
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def format_input(value: Any) -> str:
    if isinstance(value, dict) and "path" in value:
        checksum = f" sha256=`{short_hash(value['sha256'])}`" if value.get("sha256") else ""
        size = f" size=`{value['size_bytes']}`" if value.get("size_bytes") is not None else ""
        return f"`{value['path']}` exists=`{value['exists']}`{size}{checksum}"
    if isinstance(value, list):
        return ", ".join(format_input(item) for item in value) or "`[]`"
    return f"`{value}`"


def short_hash(value: str) -> str:
    return value[:12]
