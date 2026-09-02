from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from flightrl.sixdof.command_replay import action_from_command_row, desired_velocity_world, replay_velocity_commands
from flightrl.sixdof.env import SixDofEnv


ROOT = Path(__file__).resolve().parents[1]


def test_action_from_command_row_converts_body_velocity_to_low_level_action() -> None:
    env = SixDofEnv(num_envs=1, seed=3)
    row = base_row(vx="0.3", vy="0.0", vz="0.1", yawrate="30.0")

    action = action_from_command_row(env, row, velocity_gain=2.0, yawrate_scale=2.0)

    assert action.shape == (1, 4)
    assert action[0, 0] > 0.0
    assert action[0, 2] > 0.0
    assert action[0, 3] > 0.0


def test_desired_velocity_world_supports_body_world_and_sign_conventions() -> None:
    body = desired_velocity_world(1.0, 0.0, 0.2, np.pi / 2.0, "body")
    world = desired_velocity_world(1.0, 0.0, 0.2, np.pi / 2.0, "world")

    assert np.allclose(body, [0.0, 1.0, 0.2], atol=1e-6)
    assert np.allclose(world, [1.0, 0.0, 0.2], atol=1e-6)


def test_replay_velocity_commands_normalizes_origin_and_preserves_rows() -> None:
    rows = [
        base_row(t="100.0", x="4.0", y="-2.0", vx="0.1"),
        base_row(t="100.05", x="4.01", y="-2.0", vx="0.1"),
        base_row(t="100.10", x="4.02", y="-2.0", vx="0.0"),
    ]

    sim_rows, normalized = replay_velocity_commands(rows, velocity_gain=1.0)

    assert len(sim_rows) == len(rows)
    assert len(normalized) == len(rows)
    assert normalized[0]["host_time_s"] == "0.0"
    assert normalized[0]["stateEstimate.x"] == "0.0"
    assert sim_rows[0]["host_time_s"] == 0.0
    assert np.isfinite(sim_rows[-1]["stateEstimate.x"])


def test_replay_velocity_commands_can_override_unreliable_height() -> None:
    rows = [base_row(z="0.02"), base_row(t="0.05", z="0.03")]

    sim_rows, normalized = replay_velocity_commands(rows, override_z_m=0.55)

    assert normalized[0]["raw_stateEstimate.z"] == "0.02"
    assert normalized[0]["stateEstimate.z"] == "0.55"
    assert abs(sim_rows[0]["stateEstimate.z"] - 0.55) < 1e-6


def test_replay_velocity_commands_can_hold_assumed_height() -> None:
    rows = [
        base_row(z="0.02", vz="0.4"),
        base_row(t="0.05", z="0.02", vz="0.4"),
        base_row(t="0.10", z="0.02", vz="0.4"),
    ]

    sim_rows, _normalized = replay_velocity_commands(rows, override_z_m=0.55, hold_z_m=0.55)

    assert max(abs(row["stateEstimate.z"] - 0.55) for row in sim_rows) < 0.01


def test_replay_crazyflie_commands_cli_writes_sim_and_normalized_real(tmp_path: Path) -> None:
    input_path = tmp_path / "real.csv"
    sim_path = tmp_path / "sim.csv"
    real_path = tmp_path / "real.normalized.csv"
    room_path = tmp_path / "room.json"
    write_rows(input_path, [base_row(t="9.0"), base_row(t="9.05", x="0.01")])
    room_path.write_text(json.dumps({"room_estimate": room_estimate()}) + "\n")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/replay_crazyflie_commands.py",
            "--input",
            str(input_path),
            "--room-report",
            str(room_path),
            "--output",
            str(sim_path),
            "--normalized-real-output",
            str(real_path),
            "--override-z-m",
            "0.55",
            "--hold-z-m",
            "0.55",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "simulated rows" in result.stdout
    assert len(list(csv.DictReader(sim_path.open()))) == 2
    real_rows = list(csv.DictReader(real_path.open()))
    assert real_rows[0]["host_time_s"] == "0.0"
    assert real_rows[0]["stateEstimate.z"] == "0.55"


def base_row(
    *,
    t: str = "0.0",
    x: str = "0.0",
    y: str = "0.0",
    z: str = "0.45",
    vx: str = "0.0",
    vy: str = "0.0",
    vz: str = "0.0",
    yawrate: str = "0.0",
) -> dict[str, str]:
    return {
        "host_time_s": t,
        "stateEstimate.x": x,
        "stateEstimate.y": y,
        "stateEstimate.z": z,
        "stateEstimate.vx": "0.0",
        "stateEstimate.vy": "0.0",
        "stateEstimate.vz": "0.0",
        "stabilizer.roll": "0.0",
        "stabilizer.pitch": "0.0",
        "stabilizer.yaw": "0.0",
        "range.front": "1000",
        "range.back": "1000",
        "range.left": "1000",
        "range.right": "1000",
        "range.up": "1000",
        "range.zrange": "450",
        "vx_m_s": vx,
        "vy_m_s": vy,
        "vz_m_s": vz,
        "yawrate_deg_s": yawrate,
    }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def room_estimate() -> dict[str, float]:
    return {
        "x_min": -2.0,
        "x_max": 2.0,
        "y_min": -2.0,
        "y_max": 2.0,
        "z_min": 0.0,
        "z_max": 2.5,
        "max_range_m": 4.0,
    }
