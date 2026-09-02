from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from flightrl.sixdof import BoxRoom, SixDofEnv


ROOT = Path(__file__).resolve().parents[1]


def test_sixdof_reset_starts_inside_custom_room() -> None:
    room = BoxRoom(x_min=-0.4, x_max=0.5, y_min=-0.3, y_max=0.6, z_min=0.0, z_max=1.0)
    env = SixDofEnv(num_envs=32, seed=7, room=room)
    env.reset(seed=7)

    assert room.contains(env.position, margin=0.03).all()


def test_native_step_env_matches_python_with_custom_room() -> None:
    room = BoxRoom(x_min=-0.7, x_max=0.9, y_min=-0.8, y_max=0.6, z_min=0.0, z_max=1.4, max_range_m=2.0)
    python_env = SixDofEnv(num_envs=8, seed=9, room=room, use_native_step=False)
    native_env = SixDofEnv(num_envs=8, seed=9, room=room, use_native_step=True)
    actions = [[0.02, 0.01, -0.02, 0.03]] * 8

    python_env.step(actions)
    native_env.step(actions)

    assert abs(float(python_env.ranges_m[0, 0] - native_env.ranges_m[0, 0])) < 1e-5
    assert (python_env.terminals == native_env.terminals).all()


def test_sixdof_rollout_can_use_room_estimate_report(tmp_path: Path) -> None:
    room_report = tmp_path / "room.json"
    room_report.write_text(json.dumps({"room_estimate": room_estimate()}) + "\n")
    rollout = tmp_path / "rollout.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "rollout_sixdof_policy.py"),
            "--teacher",
            "--room-report",
            str(room_report),
            "--steps",
            "3",
            "--output",
            str(rollout),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    with rollout.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert max(float(row["range.front"]) for row in rows) <= 6000.0


def test_sixdof_rollout_can_use_room_report_with_native_step(tmp_path: Path) -> None:
    room_report = tmp_path / "room.json"
    room_report.write_text(json.dumps({"room_estimate": room_estimate()}) + "\n")
    rollout = tmp_path / "rollout.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "rollout_sixdof_policy.py"),
            "--teacher",
            "--room-report",
            str(room_report),
            "--native-step",
            "--steps", "3",
            "--output", str(rollout),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    with rollout.open() as handle:
        assert len(list(csv.DictReader(handle))) == 3


def room_estimate() -> dict[str, float]:
    return {
        "x_min": -3.0,
        "x_max": 3.0,
        "y_min": -2.5,
        "y_max": 2.5,
        "z_min": 0.0,
        "z_max": 3.0,
        "max_range_m": 6.0,
    }
