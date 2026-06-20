from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from flightrl.hardware.ranger_map import points_from_rows
from flightrl.sixdof import BoxRoom, SixDofCrazyflieEnv, SixDofPolicy, native_step, teacher_actions
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


def test_sixdof_reset_done_only_resets_done_rows() -> None:
    env = SixDofCrazyflieEnv(num_envs=4, seed=9)
    env.reset(seed=9)
    before = env.position.copy()
    obs = env.reset_done(np.asarray([False, True, False, True]))
    assert obs.shape == (4, 28)
    assert np.allclose(env.position[[0, 2]], before[[0, 2]])
    assert not np.allclose(env.position[[1, 3]], before[[1, 3]])


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


def test_obstacle_teacher_responds_to_vertical_clearance() -> None:
    env = SixDofCrazyflieEnv(num_envs=1, seed=6, task="obstacle_avoidance")
    env.position[:] = np.asarray([[0.0, 0.0, 0.5]], dtype=np.float32)
    env.target_position[:] = env.position
    env.velocity[:] = 0.0
    env.body_rates[:] = 0.0
    env.quaternion[:] = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    env.ranges_m[:] = np.asarray([[2.0, 2.0, 2.0, 2.0, 2.0, 0.5]], dtype=np.float32)
    open_action = teacher_actions(env, task="obstacle_avoidance")[0]

    env.ranges_m[0, 5] = 0.16
    bottom_action = teacher_actions(env, task="obstacle_avoidance")[0]
    env.ranges_m[0, 5] = 0.5
    env.ranges_m[0, 4] = 0.18
    top_action = teacher_actions(env, task="obstacle_avoidance")[0]

    assert bottom_action[0] > open_action[0]
    assert top_action[0] < open_action[0]


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
            "--eval-steps",
            "4",
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


def test_sixdof_multitask_training_and_rollout_smoke(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sixdof_multitask.pt"
    train = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_sixdof_teacher.py"),
            "--task",
            "position_yaw,obstacle_avoidance",
            "--updates",
            "1",
            "--steps-per-update",
            "2",
            "--num-envs",
            "8",
            "--batch-size",
            "16",
            "--eval-steps",
            "4",
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert checkpoint.exists()
    assert "per_task" in train.stdout

    rollout = tmp_path / "multitask_rollout.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "rollout_sixdof_policy.py"),
            "--checkpoint",
            str(checkpoint),
            "--task",
            "obstacle_avoidance",
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
    assert "action_thrust" in rows[0]

    report = tmp_path / "multitask_eval.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_sixdof_checkpoint.py"),
            "--checkpoint",
            str(checkpoint),
            "--steps",
            "4",
            "--num-envs",
            "8",
            "--output",
            str(report),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    evaluation = json.loads(report.read_text())
    assert "gate" in evaluation
    assert "teacher_action_l2_mean" in evaluation["metrics"]
    assert "action_saturation_fraction" in evaluation["metrics"]

    teacher_report = tmp_path / "teacher_eval.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_sixdof_checkpoint.py"),
            "--teacher",
            "--task",
            "position_yaw,obstacle_avoidance",
            "--steps",
            "4",
            "--num-envs",
            "8",
            "--output",
            str(teacher_report),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    teacher_evaluation = json.loads(teacher_report.read_text())
    assert teacher_evaluation["controller"] == "teacher"
    assert "teacher_action_l2_mean" not in teacher_evaluation["metrics"]
    assert "action_saturation_fraction" in teacher_evaluation["metrics"]


def test_history_observation_checkpoint_rollout_smoke(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sixdof_history.pt"
    torch.save(
        {
            "state_dict": SixDofPolicy(hidden_size=16, input_dim=60).state_dict(),
            "hidden_size": 16,
            "observation_dim": 60,
            "observation_mode": "history1",
            "task": "position_yaw",
            "tasks": ["position_yaw"],
        },
        checkpoint,
    )
    rollout = tmp_path / "history_rollout.csv"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "rollout_sixdof_policy.py"), "--checkpoint", str(checkpoint), "--steps", "3", "--output", str(rollout)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    with rollout.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert "action_thrust" in rows[0]


def test_native_sixdof_step_matches_python_step() -> None:
    env = SixDofCrazyflieEnv(num_envs=16, seed=31)
    env.reset(seed=31)
    actions = teacher_actions(env, task="position_yaw").astype(np.float32)
    position = env.position.copy()
    velocity = env.velocity.copy()
    quaternion = env.quaternion.copy()
    body_rates = env.body_rates.copy()
    ranges = env.ranges_m.copy()

    env.step(actions)
    native_step(position, velocity, quaternion, body_rates, ranges, actions, env.dt)

    assert np.allclose(position, env.position, atol=1e-7)
    assert np.allclose(velocity, env.velocity, atol=1e-6)
    assert np.allclose(quaternion, env.quaternion, atol=1e-6)
    assert np.allclose(body_rates, env.body_rates, atol=1e-6)
    assert np.allclose(ranges, env.ranges_m, atol=1e-5)


def test_native_step_env_matches_python_env_rollout() -> None:
    python_env = SixDofCrazyflieEnv(num_envs=8, seed=41, use_native_step=False)
    native_env = SixDofCrazyflieEnv(num_envs=8, seed=41, use_native_step=True)
    obs_py, _ = python_env.reset(seed=41)
    obs_native, _ = native_env.reset(seed=41)
    assert np.allclose(obs_py, obs_native)

    rng = np.random.default_rng(41)
    for _ in range(12):
        actions = rng.uniform(-0.25, 0.25, size=(8, 4)).astype(np.float32)
        obs_py, rewards_py, terminals_py, truncations_py, _ = python_env.step(actions)
        obs_native, rewards_native, terminals_native, truncations_native, _ = native_env.step(actions)
        assert np.allclose(obs_py, obs_native, atol=1e-5)
        assert np.allclose(rewards_py, rewards_native, atol=1e-5)
        assert np.array_equal(terminals_py, terminals_native)
        assert np.array_equal(truncations_py, truncations_native)
