from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ranger_log_imitation_training_writes_checkpoint_and_report(tmp_path: Path) -> None:
    log = tmp_path / "avoidance.csv"
    log.write_text(
        "range.front,range.back,range.left,range.right,range.up,range.zrange,vx_m_s,vy_m_s,yawrate_deg_s,zdistance_m\n"
        "500,2000,2000,2000,2000,500,-0.3,0.0,0.0,0.5\n"
        "2000,500,2000,2000,2000,500,0.3,0.0,0.0,0.5\n"
        "2000,2000,400,2000,2000,500,0.0,-0.3,0.0,0.5\n"
        "2000,2000,2000,400,2000,500,0.0,0.3,0.0,0.5\n"
    )
    checkpoint = tmp_path / "checkpoint.pt"
    report = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_ranger_avoidance_from_logs.py",
            "--input",
            str(log),
            "--checkpoint",
            str(checkpoint),
            "--report",
            str(report),
            "--epochs",
            "2",
            "--hidden-size",
            "8",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    data = json.loads(report.read_text())
    assert checkpoint.exists()
    assert report.with_suffix(".md").exists()
    assert data["train_samples"] == 4
    assert "val_loss=" in result.stdout
