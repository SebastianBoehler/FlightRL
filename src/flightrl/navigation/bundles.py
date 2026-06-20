from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flightrl.sim2real.provenance import path_provenance


def build_candidate_bundle(
    *,
    name: str,
    checkpoint: Path,
    benchmark_report: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    benchmark = read_json(benchmark_report)
    eligibility, blocking_reasons = eligibility_from_benchmark(benchmark)
    manifest_path = output_dir / f"{name}_bundle.json"
    markdown_path = output_dir / f"{name}_bundle.md"
    bundle = {
        "name": name,
        "checkpoint": str(checkpoint),
        "benchmark_report": str(benchmark_report),
        "inputs": {
            "checkpoint": path_provenance(checkpoint),
            "benchmark_report": path_provenance(benchmark_report),
        },
        "hardware_eligibility": eligibility,
        "blocking_reasons": blocking_reasons,
        "benchmark_summary": benchmark.get("summary", {}),
        "schemas": {
            "observation": "range_telemetry",
            "action": "firmware_setpoint",
            "mission": "single_drone_state_machine_v1",
        },
        "future_extension": {
            "multi_agent_ready": True,
            "implemented_multi_agent": False,
            "agent_id_field": "agent_id",
        },
        "files": {
            "manifest": str(manifest_path),
            "markdown": str(markdown_path),
        },
        "safety": "Candidate bundle is for reproducible simulation and shadow review; it does not approve live flight.",
    }
    write_bundle(bundle, manifest_path, markdown_path)
    return bundle


def eligibility_from_benchmark(benchmark: dict[str, Any]) -> tuple[str, list[str]]:
    summary = benchmark.get("summary", {})
    if int(summary.get("total_records", 0)) <= 0:
        return "blocked", ["benchmark_empty"]
    if not bool(summary.get("all_passed", False)):
        return "blocked", ["benchmark_failures"]
    return "shadow_only", []


def render_markdown(bundle: dict[str, Any]) -> str:
    lines = [
        f"# Navigation Candidate Bundle: {bundle['name']}",
        "",
        f"- Hardware eligibility: `{bundle['hardware_eligibility']}`",
        f"- Checkpoint: `{bundle['checkpoint']}`",
        f"- Benchmark report: `{bundle['benchmark_report']}`",
        f"- Observation schema: `{bundle['schemas']['observation']}`",
        f"- Action schema: `{bundle['schemas']['action']}`",
        f"- Blocking reasons: `{', '.join(bundle['blocking_reasons']) or 'none'}`",
        "",
        bundle["safety"],
    ]
    return "\n".join(lines)


def write_bundle(bundle: dict[str, Any], manifest_path: Path, markdown_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_markdown(bundle) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
