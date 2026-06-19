from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from flightrl.hardware.avoidance_policy import RangerReading
from flightrl.hardware.target_conditioned_policy import TargetConditionedPolicy, TargetSpec, command_from_target_model, target_observation


ROOT = Path(__file__).resolve().parents[1]


def test_target_observation_includes_direction_and_speed() -> None:
    observation = target_observation(RangerReading(1.0, 1.0, 1.0, 1.0, 2.0, 0.5), TargetSpec(90.0, 0.22), max_speed_m_s=1.1)

    assert observation.shape == (9,)
    assert abs(observation[-3]) < 1e-6
    assert np.isclose(observation[-2], 1.0)
    assert np.isclose(observation[-1], 0.2)


def test_target_model_command_clips_speed() -> None:
    model = TargetConditionedPolicy(hidden_size=8)
    for parameter in model.parameters():
        parameter.data.zero_()
    model.net[-1].bias.data = torch.tensor([2.0, -2.0, 0.0, 0.5])

    command = command_from_target_model(model, RangerReading(1.0, 1.0, 1.0, 1.0, 2.0, 0.5), TargetSpec(0.0, 0.2), max_speed_m_s=0.7)

    assert command.vx_m_s == 0.7
    assert command.vy_m_s == -0.7


def test_target_conditioned_train_and_eval_cli(tmp_path: Path) -> None:
    log = tmp_path / "target.csv"
    log.write_text(
        "range.front,range.back,range.left,range.right,range.up,range.zrange,vx_m_s,vy_m_s,yawrate_deg_s,zdistance_m\n"
        "2000,2000,2000,2000,2000,500,0.1,0.0,0.0,0.5\n"
        "600,2000,2000,2000,2000,500,-0.2,0.0,0.0,0.5\n"
        "2000,2000,600,2000,2000,500,0.0,-0.2,0.0,0.5\n"
        "2000,2000,2000,600,2000,500,0.0,0.2,0.0,0.5\n"
    )
    checkpoint = tmp_path / "target.pt"
    report = tmp_path / "train.json"
    spec = f"{log},0,0.16"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_target_conditioned_avoidance.py",
            "--input",
            spec,
            "--checkpoint",
            str(checkpoint),
            "--report",
            str(report),
            "--epochs",
            "2",
            "--hidden-size",
            "8",
            "--synthetic-close-samples",
            "4",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    eval_report = tmp_path / "eval.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_target_conditioned_avoidance.py",
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(log),
            "--target-direction-deg",
            "0",
            "--target-speed-m-s",
            "0.16",
            "--output",
            str(eval_report),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert checkpoint.exists()
    assert json.loads(eval_report.read_text())["samples"] == 4
