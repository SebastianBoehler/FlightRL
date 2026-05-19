from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys

import numpy as np

from flightrl.hardware.ranger_map import points_from_rows
from flightrl.sixdof import BoxRoom, SixDofCrazyflieEnv, teacher_actions
from flightrl.sixdof.geometry import body_rays_world


ROOT = Path(__file__).resolve().parents[1]


def test_sixdof_env_shapes_and_teacher_step() -> None:
    env = SixDofCrazyflieEnv(num_envs=4, seed=5)
    obs, _ = env.reset(seed=5)
    assert obs.shape == (4, 28)
    actions = teacher_actions(env, task="position_yaw")
    next_obs, rewards, terminals, truncations, _info = env.step(actions)
    assert actions.shape == (4, 4)
    assert next_obs.shape == obs.shape
    assert rewards.shape == (4,)
    assert terminals.shape == (4,)
    assert truncations.shape == (4,)


def test_box_room_raycast_matches_axis_aligned_distance() -> None:
    room = BoxRoom(x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, z_min=0.0, z_max=2.0)
    position = np.asarray([[0.25, 0.0, 0.5]], dtype=np.float32)
    direction = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    assert np.allclose(room.raycast(position, direction), [0.75])


def test_identity_body_rays_point_along_body_axes() -> None:
    quaternions = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    rays = body_rays_world(quaternions)
    assert np.allclose(rays[0, 0], [1.0, 0.0, 0.0])
    assert np.allclose(rays[0, 5], [0.0, 0.0, -1.0])


def test_ranger_map_projects_rows_to_points() -> None:
    rows = [
        {
            "host_time_s": "0.0",
            "stateEstimate.x": "1.0",
            "stateEstimate.y": "2.0",
            "stateEstimate.z": "0.5",
            "stabilizer.roll": "0.0",
            "stabilizer.pitch": "0.0",
            "stabilizer.yaw": "0.0",
            "range.front": "1000",
            "range.back": "32766",
            "range.left": "32766",
            "range.right": "32766",
            "range.up": "32766",
            "range.zrange": "500",
        }
    ]
    points = points_from_rows(rows)
    assert len(points) == 2
    assert points[0].x_m == 2.0
    assert points[1].z_m == 0.0


def test_sixdof_training_and_rollout_smoke(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sixdof.pt"
    train = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_sixdof_teacher.py"),
            "--task",
            "position_yaw",
            "--updates",
            "1",
            "--steps-per-update",
            "2",
            "--num-envs",
            "8",
            "--batch-size",
            "16",
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert checkpoint.exists()
    assert "checkpoint=" in train.stdout

    rollout = tmp_path / "rollout.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "rollout_sixdof_policy.py"),
            "--checkpoint",
            str(checkpoint),
            "--steps",
            "4",
            "--output",
            str(rollout),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    with rollout.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert "stateEstimate.x" in rows[0]
