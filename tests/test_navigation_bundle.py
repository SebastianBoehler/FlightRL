from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from flightrl.navigation.bundles import build_candidate_bundle


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_bundle_marks_passing_benchmark_as_shadow_only(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    benchmark = write_json(tmp_path / "benchmark.json", benchmark_report(passed=True))

    bundle = build_candidate_bundle(
        name="candidate",
        checkpoint=checkpoint,
        benchmark_report=benchmark,
        output_dir=tmp_path / "bundle",
    )

    assert bundle["name"] == "candidate"
    assert bundle["hardware_eligibility"] == "shadow_only"
    assert bundle["schemas"]["observation"] == "range_telemetry"
    assert bundle["schemas"]["action"] == "firmware_setpoint"
    assert bundle["future_extension"]["multi_agent_ready"] is True
    assert bundle["future_extension"]["implemented_multi_agent"] is False
    assert Path(bundle["files"]["manifest"]).exists()
    assert Path(bundle["files"]["markdown"]).exists()


def test_candidate_bundle_blocks_failed_benchmark(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    benchmark = write_json(tmp_path / "benchmark.json", benchmark_report(passed=False))

    bundle = build_candidate_bundle(
        name="candidate",
        checkpoint=checkpoint,
        benchmark_report=benchmark,
        output_dir=tmp_path / "bundle",
    )

    assert bundle["hardware_eligibility"] == "blocked"
    assert bundle["blocking_reasons"] == ["benchmark_failures"]


def test_candidate_bundle_cli_writes_manifest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    benchmark = write_json(tmp_path / "benchmark.json", benchmark_report(passed=True))
    output_dir = tmp_path / "bundle"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_navigation_candidate_bundle.py"),
            "--name",
            "candidate",
            "--checkpoint",
            str(checkpoint),
            "--benchmark",
            str(benchmark),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads((output_dir / "candidate_bundle.json").read_text())
    assert "candidate_bundle=" in result.stdout
    assert manifest["hardware_eligibility"] == "shadow_only"


def benchmark_report(*, passed: bool) -> dict:
    return {
        "summary": {
            "total_records": 1,
            "passed_records": 1 if passed else 0,
            "blocked_records": 0 if passed else 1,
            "all_passed": passed,
        },
        "records": [
            {
                "label": "candidate",
                "scenario": "target_approach",
                "passed": passed,
                "failures": [] if passed else ["mean_completed_fraction_lt_0.90"],
                "score": 0.9 if passed else 0.2,
                "checkpoint": "candidate.pt",
            }
        ],
    }


def write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data))
    return path
