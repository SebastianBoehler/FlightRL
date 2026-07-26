from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import torch

from flightrl.sixdof.mode_conditioned import ModeConditionedWrapper, append_mode_torch, expand_policy_for_modes
from flightrl.sixdof.puffer_policy import PufferPolicyMetadata, PufferSixDofPolicy


ROOT = Path(__file__).resolve().parents[1]


def tiny_policy() -> PufferSixDofPolicy:
    return PufferSixDofPolicy(PufferPolicyMetadata(28, 16, 4, 1))


def velocity_row() -> dict[str, float]:
    return {
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


def test_expand_policy_preserves_base_output_with_zero_mode_columns() -> None:
    base = tiny_policy()
    expanded = expand_policy_for_modes(base)
    observations = torch.randn(3, 28)

    assert expanded.metadata.observation_dim == 30
    assert torch.allclose(base(observations), expanded(append_mode_torch(observations, "obstacle_hover")))
    assert torch.allclose(base(observations), expanded(append_mode_torch(observations, "velocity_target")))


def test_mode_conditioned_wrapper_accepts_base_observations() -> None:
    wrapper = ModeConditionedWrapper(expand_policy_for_modes(tiny_policy()), "velocity_target")
    output = wrapper(torch.zeros(2, 28))

    assert wrapper.metadata.observation_dim == 28
    assert output.shape == (2, 4)


def test_mode_conditioned_report_cli_smoke(tmp_path: Path) -> None:
    checkpoint = tmp_path / "mode.bin"
    log = tmp_path / "velocity.csv"
    output = tmp_path / "report.json"
    torch.save(expand_policy_for_modes(tiny_policy()).state_dict(), checkpoint)
    with log.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(velocity_row().keys()))
        writer.writeheader()
        for _ in range(4):
            writer.writerow(velocity_row())

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_puffer_mode_conditioned_transfer_report.py"),
            "--candidate",
            f"tiny:{checkpoint}",
            "--velocity-live-log",
            f"smoke:{log}",
            "--output",
            str(output),
            "--steps",
            "2",
            "--num-envs",
            "4",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "mode_conditioned_transfer_report=" in result.stdout
    assert output.exists()
    assert output.with_suffix(".md").exists()
