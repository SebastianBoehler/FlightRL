from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_puffer_native_optimization_report_cli_writes_comparison(tmp_path: Path) -> None:
    native = write_json(
        tmp_path / "native.json",
        {"best": {"native_env_steps_per_second": 12_000_000, "python_steps_per_second": 1_000_000, "num_envs": 1024}},
    )
    training = write_json(
        tmp_path / "training.json",
        {"summary": {"best_total_sps": {"name": "smoke", "total_sps": 42_000, "num_envs": 64, "horizon": 16}}},
    )
    output = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_puffer_native_optimization_report.py",
            "--native-benchmark",
            str(native),
            "--training-throughput",
            str(training),
            "--baseline-native-sps",
            "6000000",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text())
    assert "optimization_report=" in result.stdout
    assert report["passed"] is True
    assert report["comparison"]["flightrl"]["observation_dim"] == 28
    assert report["throughput"]["native_speedup_vs_baseline"] == 2.0
    assert output.with_suffix(".md").exists()


def write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data))
    return path
