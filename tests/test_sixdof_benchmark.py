from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_sixdof_benchmark_sweep_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "bench.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "benchmark_sixdof_sweep.py"),
            "--env-counts",
            "4",
            "--steps",
            "2",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["best"]["num_envs"] == 4
    assert output.with_suffix(".md").exists()
