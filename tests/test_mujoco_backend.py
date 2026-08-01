from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from flightrl.mujoco import (
    MuJoCoCrazyflieEnv,
    is_mujoco_available,
    is_mujoco_rendering_available,
)
from flightrl.mujoco.control import (
    MuJoCoControlParams,
    control_from_physics_profile,
    resolve_control,
    step_actuator_targets,
)
from flightrl.mujoco.model import build_crazyflie_mjcf
from flightrl.sixdof.env import euler_to_quat
from flightrl.sixdof.geometry import AxisAlignedObstacle, BoxRoom
from flightrl.sixdof.physics import SixDofPhysicsProfile


ROOT = Path(__file__).resolve().parents[1]


def test_mujoco_model_applies_plain_room_and_physics_profile() -> None:
    room = BoxRoom(
        x_min=-0.8,
        x_max=1.2,
        y_min=-0.6,
        y_max=0.9,
        z_min=0.1,
        z_max=1.7,
        obstacles=(
            AxisAlignedObstacle(
                x_min=0.2,
                x_max=0.4,
                y_min=-0.1,
                y_max=0.3,
                z_min=0.1,
                z_max=0.8,
            ),
        ),
    )
    physics = SixDofPhysicsProfile(mass_kg=0.042, gravity_m_s2=9.7)

    root = ET.fromstring(
        build_crazyflie_mjcf(room=room, physics_profile=physics)
    )

    assert root.find("option").attrib["gravity"] == "0 0 -9.7"
    inertial = root.find(".//body[@name='crazyflie']/inertial")
    wall = root.find(".//geom[@name='wall_x_pos']")
    assert inertial is not None
    assert wall is not None
    assert inertial.attrib["mass"] == "0.042"
    assert wall.attrib["pos"] == "1.22 0.15 0.9"
    obstacle = root.find(".//geom[@name='room_obstacle_0']")
    assert obstacle is not None
    assert obstacle.attrib["pos"] == "0.3 0.1 0.45"
    assert obstacle.attrib["size"] == "0.1 0.2 0.35"


def test_mujoco_backend_shapes_when_available() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    env = MuJoCoCrazyflieEnv(num_envs=2, seed=4)
    obs, _ = env.reset(seed=4)
    assert obs.shape == (2, 28)
    actions = np.zeros((2, 4), dtype=np.float32)
    next_obs, rewards, terminals, truncations, _ = env.step(actions)
    assert next_obs.shape == obs.shape
    assert rewards.shape == (2,)
    assert terminals.shape == (2,)
    assert truncations.shape == (2,)
    assert np.isfinite(next_obs).all()


def test_plain_obstacle_room_rejects_infeasible_reset_region() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    room = BoxRoom(
        obstacles=(
            AxisAlignedObstacle(
                x_min=-1.0,
                x_max=1.0,
                y_min=-1.0,
                y_max=1.0,
                z_min=0.1,
                z_max=1.5,
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="reset poses and targets"):
        MuJoCoCrazyflieEnv(num_envs=8, seed=123, room=room)


def test_mujoco_backend_accepts_physics_profile_when_available(tmp_path: Path) -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    physics = tmp_path / "physics.json"
    physics.write_text(json.dumps({"physics_profile": {"mass_kg": 0.042, "gravity_m_s2": 9.7, "thrust_scale": 0.6}}))
    env = MuJoCoCrazyflieEnv(num_envs=1, seed=4, physics_profile=str(physics))

    assert env.control.mass_kg == pytest.approx(0.042)
    assert env.control.gravity == pytest.approx(9.7)
    assert env.control.thrust_scale == pytest.approx(0.6)


def test_mujoco_control_carries_both_physics_profile_lags() -> None:
    profile = SixDofPhysicsProfile(rate_tau_s=0.08, motor_tau_s=0.16)
    control = control_from_physics_profile(profile)

    assert control.rate_tau_s == pytest.approx(0.08)
    assert control.motor_tau_s == pytest.approx(0.16)

    thrust, rates = step_actuator_targets(
        1.0,
        np.zeros(3, dtype=np.float64),
        np.asarray([1.0, 1.0, -1.0, 0.5], dtype=np.float64),
        control,
        0.01,
    )
    assert thrust == pytest.approx(1.0 + (0.01 / 0.17) * 0.75)
    assert np.allclose(
        rates,
        (0.01 / 0.09) * np.asarray([6.0, -6.0, 2.0]),
    )


def test_mujoco_control_rejects_split_physical_profiles() -> None:
    profile = SixDofPhysicsProfile()
    conflicting = MuJoCoControlParams(mass_kg=0.040)

    with pytest.raises(ValueError, match="mass_kg"):
        resolve_control(conflicting, profile)

    tuned = MuJoCoControlParams(rate_kp=3.0e-4)
    assert resolve_control(tuned, profile) is tuned


def test_mujoco_step_updates_profile_actuator_states_when_available() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    profile = SixDofPhysicsProfile(rate_tau_s=0.09, motor_tau_s=0.19)
    env = MuJoCoCrazyflieEnv(num_envs=1, seed=4, physics_profile=profile)

    env.step(np.ones((1, 4), dtype=np.float32))

    assert env.thrust_state[0] == pytest.approx(1.0 + (0.01 / 0.20) * 0.75)
    assert np.allclose(
        env.rate_command_state[0],
        (0.01 / 0.10) * np.asarray([6.0, 6.0, 4.0]),
    )


def test_mujoco_circle_uses_shared_yaw_and_orbit_reward_when_available() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    env = MuJoCoCrazyflieEnv(num_envs=1, seed=4, task="circle")
    env.position[:] = np.asarray([[0.75, 0.0, 0.65]], dtype=np.float32)
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.65]], dtype=np.float32)
    env.target_yaw[:] = 0.0
    env.quaternion[:] = euler_to_quat(
        np.zeros(1),
        np.zeros(1),
        np.asarray([np.pi / 2.0], dtype=np.float32),
    )
    env.velocity[:] = 0.0
    env.ranges_m[:] = env.room.max_range_m
    actions = np.zeros((1, 4), dtype=np.float32)

    circle_obs = env.observation()
    circle_reward = env._reward(actions)
    env.native_task_ids[:] = 0
    position_reward = env._reward(actions)

    assert abs(circle_obs[0, 16]) < 1e-5
    assert abs(circle_obs[0, 17] - 1.0) < 1e-5
    assert circle_reward[0] > position_reward[0] + 0.8


def test_mujoco_aideck_camera_matches_gray4_contract_when_available() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    if not is_mujoco_rendering_available():
        pytest.skip("MuJoCo rendering backend is unavailable")
    env = MuJoCoCrazyflieEnv(num_envs=1, seed=4)

    first = env.render_aideck_gray4()
    env.data[0].qpos[0] += 0.5
    env.mujoco.mj_forward(env.model, env.data[0])
    translated = env.render_aideck_gray4()

    assert first.shape == (48, 64)
    assert first.dtype == np.uint8
    assert np.all(first % 17 == 0)
    assert not np.array_equal(first, translated)


def test_mujoco_benchmark_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "mujoco_bench.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "benchmark_mujoco_sixdof.py"),
            "--env-counts",
            "1",
            "--steps",
            "2",
            "--output",
            str(output),
            "--allow-missing-mujoco",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["status"] in {"ok", "missing_mujoco"}
    if report["status"] == "ok":
        assert report["results"][0]["num_envs"] == 1
        assert report["results"][0]["mujoco_sps"] > 0.0
    assert output.with_suffix(".md").exists()


def test_mujoco_checkpoint_evaluator_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "mujoco_eval.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_mujoco_sixdof_checkpoint.py"),
            "--teacher",
            "--task",
            "obstacle_avoidance",
            "--steps",
            "2",
            "--num-envs",
            "1",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["status"] in {"ok", "missing_mujoco"}
    if report["status"] == "ok":
        assert report["backend"] == "mujoco"
        assert "metrics" in report
    assert output.with_suffix(".md").exists()
