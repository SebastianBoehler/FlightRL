from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from flightrl.hardware.avoidance_policy import RangerAvoidancePolicy


ROOT = Path(__file__).resolve().parents[1]


def test_shadow_evaluator_writes_action_gap_report(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ranger.pt"
    model = RangerAvoidancePolicy(hidden_size=8)
    for parameter in model.parameters():
        parameter.data.zero_()
    model.net[-1].bias.data = torch.tensor([0.2, -0.1, 0.0, 0.5])
    torch.save({"state_dict": model.state_dict(), "hidden_size": 8}, checkpoint)

    log = tmp_path / "live.csv"
    log.write_text(
        "range.front,range.back,range.left,range.right,range.up,range.zrange,vx_m_s,vy_m_s,yawrate_deg_s,zdistance_m\n"
        "500,2000,2000,2000,2000,500,0.1,-0.1,0.0,0.5\n"
    )
    output = tmp_path / "shadow.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_ranger_avoidance_shadow.py",
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(log),
            "--output",
            str(output),
            "--max-speed-m-s",
            "1.1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text())
    assert "samples=1 passed=True" in result.stdout
    assert report["samples"] == 1
    assert report["mae"]["vx_m_s"] > 0.0
    assert report["speed"]["shadow_max_m_s"] > 0.0
    assert output.with_suffix(".md").exists()
