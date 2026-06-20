from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from flightrl.hardware.avoidance_policy import RangerReading
from flightrl.hardware.ttc_policy import TTCAvoidancePolicy, command_from_ttc_model, ttc_observation, ttc_urgency


ROOT = Path(__file__).resolve().parents[1]


def test_ttc_observation_includes_rates_and_urgency() -> None:
    reading = RangerReading(front_m=0.8, back_m=2.0, left_m=2.0, right_m=2.0, up_m=3.0, zrange_m=0.5)
    rate = RangerReading(front_m=-2.0, back_m=0.0, left_m=0.0, right_m=0.0, up_m=0.0, zrange_m=0.0)

    observation = ttc_observation(reading, rate, max_rate_m_s=4.0, ttc_horizon_s=0.7)

    assert observation.shape == (14,)
    assert np.isclose(observation[6], -0.5)
    assert observation[-1] > 0.4


def test_ttc_model_command_clips_horizontal_norm() -> None:
    model = TTCAvoidancePolicy(hidden_size=8)
    for parameter in model.parameters():
        parameter.data.zero_()
    model.net[-1].bias.data = torch.tensor([2.0, 2.0, 0.0, 0.5])

    command = command_from_ttc_model(
        model,
        RangerReading(1.0, 1.0, 1.0, 1.0, 2.0, 0.5),
        RangerReading(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        max_speed_m_s=0.6,
    )

    assert np.linalg.norm([command.vx_m_s, command.vy_m_s]) <= 0.600001


def test_ttc_urgency_is_zero_for_open_space() -> None:
    assert ttc_urgency(float("inf"), 0.7) == 0.0
    assert ttc_urgency(0.8, 0.7) == 0.0
    assert ttc_urgency(0.0, 0.7) == 1.0


def test_ttc_log_imitation_training_writes_checkpoint_and_report(tmp_path: Path) -> None:
    log = tmp_path / "ttc.csv"
    log.write_text(
        "range.front,range.back,range.left,range.right,range.up,range.zrange,"
        "range_rate_front_m_s,range_rate_back_m_s,range_rate_left_m_s,range_rate_right_m_s,range_rate_up_m_s,range_rate_zrange_m_s,"
        "min_horizontal_range_m,min_horizontal_ttc_s,vx_m_s,vy_m_s,yawrate_deg_s,zdistance_m\n"
        "500,2000,2000,2000,2000,500,-1.0,0.0,0.0,0.0,0.0,0.0,0.5,0.5,-0.3,0.0,0.0,0.5\n"
        "2000,500,2000,2000,2000,500,0.0,-1.0,0.0,0.0,0.0,0.0,0.5,0.5,0.3,0.0,0.0,0.5\n"
        "2000,2000,500,2000,2000,500,0.0,0.0,-1.0,0.0,0.0,0.0,0.5,0.5,0.0,-0.3,0.0,0.5\n"
        "2000,2000,2000,500,2000,500,0.0,0.0,0.0,-1.0,0.0,0.0,0.5,0.5,0.0,0.3,0.0,0.5\n"
    )
    checkpoint = tmp_path / "checkpoint.pt"
    report = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_ttc_avoidance_from_logs.py",
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
            "--wandb-mode",
            "disabled",
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
