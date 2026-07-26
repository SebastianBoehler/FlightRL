from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from flightrl.mujoco import MuJoCoCrazyflieEnv, is_mujoco_available


ROOT = Path(__file__).resolve().parents[1]


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


def test_mujoco_backend_accepts_physics_profile_when_available(tmp_path: Path) -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    physics = tmp_path / "physics.json"
    physics.write_text(json.dumps({"physics_profile": {"mass_kg": 0.042, "gravity_m_s2": 9.7, "thrust_scale": 0.6}}))
    env = MuJoCoCrazyflieEnv(num_envs=1, seed=4, physics_profile=str(physics))

    assert env.control.mass_kg == pytest.approx(0.042)
    assert env.control.gravity == pytest.approx(9.7)
    assert env.control.thrust_scale == pytest.approx(0.6)


def test_mujoco_aideck_camera_matches_gray4_contract_when_available() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
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
