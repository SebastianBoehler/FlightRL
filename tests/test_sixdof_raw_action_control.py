from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flightrl.hardware.sixdof_raw_action import RawPufferActionConfig


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("raw_action_control", ROOT / "scripts" / "crazyflie_sixdof_raw_action_control.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ConstantPufferPolicy(torch.nn.Module):
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        assert observations.shape == (1, 28)
        return torch.tensor([[0.1, 0.2, -0.3, 0.4]], dtype=torch.float32)


class ObservationPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observation: torch.Tensor | None = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        self.observation = observations.detach().clone()
        return torch.zeros((1, 4), dtype=torch.float32)


def test_raw_action_control_row_marks_phase_and_raw_active_state() -> None:
    args = SimpleNamespace(height_m=0.5, target_yaw_deg=0.0)
    telemetry = MODULE.synthetic_telemetry(args)

    row = MODULE.control_row(
        ConstantPufferPolicy(),
        RawPufferActionConfig(),
        telemetry,
        args,
        np.zeros(4, dtype=np.float32),
        True,
        "startup_hover",
        False,
    )

    assert row["phase"] == "startup_hover"
    assert row["controls_drone"] is True
    assert row["raw_control_active"] is False
    assert row["raw_puffer_output"] is True
    assert row["action_thrust"] == 0.10000000149011612
    assert row["pitch_rate_deg_s"] == pytest.approx(-103.1324)
    assert row["commander_pitch_rate_deg_s"] == pytest.approx(103.1324)


def test_raw_action_control_scales_previous_action_tail() -> None:
    args = SimpleNamespace(height_m=0.5, target_yaw_deg=0.0, previous_action_observation_scale=0.25)
    policy = ObservationPolicy()

    MODULE.control_row(
        policy,
        RawPufferActionConfig(),
        MODULE.synthetic_telemetry(args),
        args,
        np.asarray([1.0, -0.5, 0.25, -1.0], dtype=np.float32),
        False,
        "replay",
        False,
    )

    assert policy.observation is not None
    assert policy.observation[0, -4:].numpy() == pytest.approx([0.25, -0.125, 0.0625, -0.25])


def test_raw_action_control_current_pose_target_removes_origin_pull() -> None:
    args = SimpleNamespace(height_m=0.5, target_yaw_deg=0.0, target_mode="current_pose")
    telemetry = {**MODULE.synthetic_telemetry(args), "stateEstimate.x": 2.0, "stateEstimate.y": -1.0}
    policy = ObservationPolicy()

    row = MODULE.control_row(
        policy,
        RawPufferActionConfig(),
        telemetry,
        args,
        np.zeros(4, dtype=np.float32),
        False,
        "replay",
        False,
    )

    assert row["target_x"] == pytest.approx(2.0)
    assert row["target_y"] == pytest.approx(-1.0)
    assert policy.observation is not None
    assert policy.observation[0, :3].numpy() == pytest.approx([0.0, 0.0, 0.0])
