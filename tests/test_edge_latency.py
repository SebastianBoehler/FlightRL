from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import torch

from flightrl.edge import export_sixdof_torchscript
from flightrl.sixdof import SixDofPolicy


ROOT = Path(__file__).resolve().parents[1]


def test_edge_latency_cli_writes_report(tmp_path: Path) -> None:
    checkpoint = tmp_path / "history.pt"
    torch.save(
        {
            "state_dict": SixDofPolicy(hidden_size=16, input_dim=60).state_dict(),
            "hidden_size": 16,
            "observation_dim": 60,
            "observation_mode": "history1",
            "task": "position_yaw",
            "tasks": ["position_yaw"],
        },
        checkpoint,
    )
    model = export_sixdof_torchscript(checkpoint, tmp_path / "history.ts", samples=4).model_path
    output = tmp_path / "latency.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "benchmark_sixdof_edge_policy.py"),
            "--checkpoint",
            str(checkpoint),
            "--torchscript",
            str(model),
            "--iterations",
            "3",
            "--warmup",
            "1",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text())
    assert report["observation"]["mode"] == "history1"
    assert report["controller"] == "policy"
    assert report["eager"]["per_sample_us"] > 0.0
    assert report["torchscript_result"]["per_sample_us"] > 0.0
    assert report["max_abs_error"] <= 1e-6
    assert output.with_suffix(".md").exists()
