from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import torch

from flightrl.desktop import export_sixdof_desktop_torchscript
from flightrl.sixdof import SixDofPolicy, build_checkpoint_payload


ROOT = Path(__file__).resolve().parents[1]


def test_desktop_latency_cli_writes_report(tmp_path: Path) -> None:
    checkpoint = tmp_path / "history.pt"
    torch.save(
        build_checkpoint_payload(
            state_dict=SixDofPolicy(hidden_size=16, input_dim=60).state_dict(),
            tasks=("position_yaw",),
            hidden_size=16,
            observation_mode="history1",
        ),
        checkpoint,
    )
    model = export_sixdof_desktop_torchscript(checkpoint, tmp_path / "history.ts", samples=4).model_path
    output = tmp_path / "latency.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "benchmark_sixdof_desktop_policy.py"),
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
    assert report["evidence_scope"] == "desktop_cpu_only"
    assert report["deployment_authority"] is False
    assert report["checkpoint"]["path"] == str(checkpoint.resolve())
    assert report["torchscript"]["path"] == str(model.resolve())
    assert report["observation"]["mode"] == "history1"
    assert report["controller"] == "policy"
    assert report["eager"]["per_sample_us"] > 0.0
    assert report["torchscript_result"]["per_sample_us"] > 0.0
    assert report["max_abs_error"] <= 1e-6
    assert output.with_suffix(".md").exists()
