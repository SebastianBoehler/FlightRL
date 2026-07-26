from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from flightrl.hardware.sixdof_puffer_shadow import PufferShadowConfig, puffer_shadow_row
from flightrl.hardware.sixdof_raw_action import RawPufferActionConfig
from flightrl.sixdof.puffer_policy import PufferPolicyMetadata, PufferSixDofPolicy


ROOT = Path(__file__).resolve().parents[1]


class ConstantPufferPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.metadata = SimpleNamespace(observation_dim=28, action_dim=4)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        assert observations.shape == (1, 28)
        return torch.tensor([[0.5, -0.25, 0.5, -1.0]], dtype=torch.float32)


class ObservationPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observation: torch.Tensor | None = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        self.observation = observations.detach().clone()
        return torch.zeros((1, 4), dtype=torch.float32)


def test_puffer_shadow_row_logs_raw_action_without_control() -> None:
    row = puffer_shadow_row(
        ConstantPufferPolicy(),
        {
            "range.front": 260.0,
            "range.back": 1800.0,
            "range.left": 900.0,
            "range.right": 900.0,
            "range.up": 1500.0,
            "range.zrange": 500.0,
            "stateEstimate.z": 0.5,
            "pm.vbat": 3.85,
        },
        PufferShadowConfig(
            raw_action=RawPufferActionConfig(
                hover_thrust_percent=48.0,
                thrust_scale=0.75,
                max_roll_rate_deg_s=300.0,
                max_pitch_rate_deg_s=400.0,
                max_yaw_rate_deg_s=200.0,
            )
        ),
        previous_action=np.zeros(4, dtype=np.float32),
    )

    assert row["monitor_only"] is True
    assert row["controls_drone"] is False
    assert row["raw_puffer_output"] is True
    assert row["thrust_percent"] == 66.0
    assert row["roll_rate_deg_s"] == -75.0
    assert row["pitch_rate_deg_s"] == 200.0
    assert row["commander_pitch_rate_deg_s"] == -200.0
    assert row["yaw_rate_deg_s"] == -200.0


def test_puffer_shadow_row_scales_previous_action_tail() -> None:
    policy = ObservationPolicy()

    puffer_shadow_row(
        policy,
        {
            "range.front": 260.0,
            "range.back": 1800.0,
            "range.left": 900.0,
            "range.right": 900.0,
            "range.up": 1500.0,
            "range.zrange": 500.0,
            "stateEstimate.z": 0.5,
        },
        PufferShadowConfig(previous_action_observation_scale=0.25),
        previous_action=np.asarray([1.0, -0.5, 0.25, -1.0], dtype=np.float32),
    )

    assert policy.observation is not None
    assert policy.observation[0, -4:].numpy().tolist() == [0.25, -0.125, 0.0625, -0.25]


def test_puffer_shadow_cli_dry_run_loads_checkpoint_without_hardware(tmp_path: Path) -> None:
    checkpoint = tmp_path / "puffer.bin"
    output = tmp_path / "shadow.csv"
    policy = PufferSixDofPolicy(PufferPolicyMetadata(observation_dim=28, hidden_size=8, action_dim=4, num_layers=2))
    torch.save(policy.state_dict(), checkpoint)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "crazyflie_sixdof_puffer_shadow_monitor.py"),
            "--checkpoint",
            str(checkpoint),
            "--dry-run",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "controls_drone=False" in result.stdout
    rows = list(csv.DictReader(output.open()))
    assert rows[0]["monitor_only"] == "True"
    assert rows[0]["controls_drone"] == "False"
    assert rows[0]["raw_puffer_output"] == "True"
