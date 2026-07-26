from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import torch

from flightrl.sixdof.puffer_policy import PufferPolicyMetadata, PufferSixDofPolicy
from flightrl.sixdof.velocity_transfer import VelocityTransferConfig, score_velocity_transfer_policy


ROOT = Path(__file__).resolve().parents[1]


class ConstantPolicy(torch.nn.Module):
    def __init__(self, action: tuple[float, float, float, float]) -> None:
        super().__init__()
        self.action = torch.tensor(action, dtype=torch.float32)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.action.repeat(observations.shape[0], 1)


def velocity_row(**overrides: float) -> dict[str, float]:
    row = {
        "stateEstimate.x": 0.0,
        "stateEstimate.y": 0.0,
        "stateEstimate.z": 0.5,
        "stateEstimate.vx": 0.0,
        "stateEstimate.vy": 0.0,
        "stateEstimate.vz": 0.0,
        "stabilizer.roll": 0.0,
        "stabilizer.pitch": 0.0,
        "stabilizer.yaw": 0.0,
        "gyro.x": 0.0,
        "gyro.y": 0.0,
        "gyro.z": 0.0,
        "range.front": 800.0,
        "range.back": 900.0,
        "range.left": 700.0,
        "range.right": 600.0,
        "range.up": 1500.0,
        "range.zrange": 500.0,
        "target_x": 0.0,
        "target_y": 0.0,
        "target_z": 0.5,
        "action_thrust": 0.0,
        "action_roll_rate": 0.0,
        "action_pitch_rate": 0.0,
        "action_yaw_rate": 0.0,
        "vx_m_s": 0.0,
        "vy_m_s": 0.0,
        "vz_m_s": 0.0,
        "yawrate_deg_s": 0.0,
    }
    row.update(overrides)
    return row


def test_velocity_transfer_gate_passes_matching_zero_command() -> None:
    report = score_velocity_transfer_policy(
        ConstantPolicy((0.0, 0.0, 0.0, 0.0)),
        [velocity_row() for _ in range(4)],
        VelocityTransferConfig(min_samples=4),
    )

    assert report["gate"]["passed"] is True
    assert report["source_adapter"]["horizontal_l2_p95_m_s"] == 0.0


def test_velocity_transfer_gate_flags_wrong_direction() -> None:
    report = score_velocity_transfer_policy(
        ConstantPolicy((0.0, 0.0, 1.0, 0.0)),
        [velocity_row(**{"vx_m_s": -0.08}) for _ in range(4)],
        VelocityTransferConfig(min_samples=4),
    )

    assert report["gate"]["passed"] is False
    assert "velocity_horizontal_l2_p95" in report["gate"]["failures"]
    assert "velocity_vx_sign" in report["gate"]["failures"]


def test_velocity_transfer_cli_smoke(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny.bin"
    log = tmp_path / "velocity.csv"
    output = tmp_path / "velocity_gate.json"
    torch.save(PufferSixDofPolicy(PufferPolicyMetadata(28, 16, 4, 1)).state_dict(), checkpoint)
    with log.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(velocity_row().keys()))
        writer.writeheader()
        for _ in range(4):
            writer.writerow(velocity_row())

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_puffer_velocity_transfer_gate.py"),
            "--candidate",
            f"tiny:{checkpoint}",
            "--live-log",
            f"smoke:{log}",
            "--output",
            str(output),
            "--min-samples",
            "4",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "puffer_velocity_transfer_gate=" in result.stdout
    assert output.exists()
    assert output.with_suffix(".md").exists()
