from __future__ import annotations

import json
from pathlib import Path

import torch

from flightrl.desktop import export_sixdof_desktop_torchscript
from flightrl.sixdof import SixDofPolicy, build_checkpoint_payload


def test_export_sixdof_desktop_torchscript_writes_parity_report(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sixdof.pt"
    torch.save(
        build_checkpoint_payload(
            state_dict=SixDofPolicy(hidden_size=32).state_dict(),
            tasks=("position_yaw",),
            hidden_size=32,
        ),
        checkpoint,
    )

    result = export_sixdof_desktop_torchscript(checkpoint, tmp_path / "sixdof.ts", samples=8)

    assert result.model_path.exists()
    assert result.report_path.exists()
    report = json.loads(result.report_path.read_text())
    assert report["evidence_scope"] == "desktop_cpu_only"
    assert report["deployment_authority"] is False
    assert report["checkpoint"]["path"] == str(checkpoint.resolve())
    assert report["model"]["path"] == str(result.model_path.resolve())
    assert report["action"]["meaning"] == ["thrust", "roll_rate", "pitch_rate", "yaw_rate"]
    assert report["parity"]["max_abs_error"] <= 1e-6

    model = torch.jit.load(str(result.model_path))
    output = model(torch.zeros((2, 28), dtype=torch.float32))
    assert output.shape == (2, 4)


def test_export_sixdof_desktop_torchscript_supports_task_conditioned_policy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sixdof_multitask.pt"
    torch.save(
        build_checkpoint_payload(
            state_dict=SixDofPolicy(hidden_size=32, input_dim=30).state_dict(),
            tasks=("position_yaw", "obstacle_avoidance"),
            hidden_size=32,
        ),
        checkpoint,
    )

    result = export_sixdof_desktop_torchscript(checkpoint, tmp_path / "sixdof_multitask.ts", samples=8)

    report = json.loads(result.report_path.read_text())
    assert report["observation"]["shape"] == [30]
    assert report["observation"]["task_conditioned"] is True
    model = torch.jit.load(str(result.model_path))
    assert model(torch.zeros((2, 30), dtype=torch.float32)).shape == (2, 4)


def test_export_sixdof_desktop_torchscript_supports_history_observation_mode(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sixdof_history.pt"
    torch.save(
        build_checkpoint_payload(
            state_dict=SixDofPolicy(hidden_size=32, input_dim=60).state_dict(),
            tasks=("position_yaw",),
            hidden_size=32,
            observation_mode="history1",
        ),
        checkpoint,
    )

    result = export_sixdof_desktop_torchscript(checkpoint, tmp_path / "sixdof_history.ts", samples=8)

    report = json.loads(result.report_path.read_text())
    assert report["observation"]["shape"] == [60]
    assert report["observation"]["mode"] == "history1"
    model = torch.jit.load(str(result.model_path))
    assert model(torch.zeros((2, 60), dtype=torch.float32)).shape == (2, 4)


def test_export_sixdof_desktop_torchscript_marks_teacher_residual_actor(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sixdof_residual.pt"
    torch.save(
        build_checkpoint_payload(
            state_dict=SixDofPolicy(hidden_size=16).state_dict(),
            tasks=("circle",),
            hidden_size=16,
            controller="teacher_residual",
            residual_scale=0.05,
        ),
        checkpoint,
    )

    result = export_sixdof_desktop_torchscript(checkpoint, tmp_path / "sixdof_residual.ts", samples=8)

    report = json.loads(result.report_path.read_text())
    assert report["controller"] == "teacher_residual"
    assert report["residual_scale"] == 0.05
    assert report["action"]["meaning"] == ["thrust_residual", "roll_rate_residual", "pitch_rate_residual", "yaw_rate_residual"]
