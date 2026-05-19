from __future__ import annotations

import json
from pathlib import Path

import torch

from flightrl.edge import export_sixdof_torchscript
from flightrl.sixdof import SixDofPolicy


def test_export_sixdof_torchscript_writes_parity_report(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sixdof.pt"
    torch.save(
        {
            "state_dict": SixDofPolicy(hidden_size=32).state_dict(),
            "hidden_size": 32,
            "task": "position_yaw",
        },
        checkpoint,
    )

    result = export_sixdof_torchscript(checkpoint, tmp_path / "sixdof.ts", samples=8)

    assert result.model_path.exists()
    assert result.report_path.exists()
    report = json.loads(result.report_path.read_text())
    assert report["action"]["meaning"] == ["thrust", "roll_rate", "pitch_rate", "yaw_rate"]
    assert report["parity"]["max_abs_error"] <= 1e-6

    model = torch.jit.load(str(result.model_path))
    output = model(torch.zeros((2, 28), dtype=torch.float32))
    assert output.shape == (2, 4)


def test_export_sixdof_torchscript_supports_task_conditioned_policy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sixdof_multitask.pt"
    torch.save(
        {
            "state_dict": SixDofPolicy(hidden_size=32, input_dim=30).state_dict(),
            "hidden_size": 32,
            "observation_dim": 30,
            "task_conditioned": True,
            "tasks": ["position_yaw", "obstacle_avoidance"],
        },
        checkpoint,
    )

    result = export_sixdof_torchscript(checkpoint, tmp_path / "sixdof_multitask.ts", samples=8)

    report = json.loads(result.report_path.read_text())
    assert report["observation"]["shape"] == [30]
    assert report["observation"]["task_conditioned"] is True
    model = torch.jit.load(str(result.model_path))
    assert model(torch.zeros((2, 30), dtype=torch.float32)).shape == (2, 4)
