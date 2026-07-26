from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from flightrl.sixdof import SixDofCrazyflieEnv


ROOT = Path(__file__).resolve().parents[1]


def test_obstacle_close_live_profile_samples_close_ranges() -> None:
    env = SixDofCrazyflieEnv(num_envs=1024, seed=42, task="obstacle_avoidance", reset_profile="obstacle_close_live")
    env.reset(seed=42)
    hmin = np.min(env.ranges_m[:, :4], axis=1)

    assert np.quantile(hmin, 0.10) < 0.25
    assert np.quantile(hmin, 0.50) < 1.2
    assert np.all(env.position[:, 2] >= 0.25)


def test_obstacle_hover_live_profile_keeps_target_at_start_pose() -> None:
    env = SixDofCrazyflieEnv(num_envs=1024, seed=44, task="obstacle_avoidance", reset_profile="obstacle_hover_live")
    env.reset(seed=44)
    hmin = np.min(env.ranges_m[:, :4], axis=1)

    np.testing.assert_allclose(env.target_position, env.position, rtol=1e-6, atol=1e-6)
    assert np.quantile(hmin, 0.10) < 0.25
    assert np.quantile(hmin, 0.50) < 1.2


def test_obstacle_vertical_live_profile_samples_top_and_bottom_ranges() -> None:
    env = SixDofCrazyflieEnv(num_envs=2048, seed=43, task="obstacle_avoidance", reset_profile="obstacle_vertical_live")
    env.reset(seed=43)

    assert np.quantile(env.ranges_m[:, 5], 0.10) < 0.35
    assert np.quantile(env.ranges_m[:, 4], 0.10) < 0.45
    assert np.all(env.position[:, 2] >= 0.12)
    assert np.all(env.position[:, 2] <= env.room.z_max - 0.12)


def test_summarize_sixdof_reset_profile_script(tmp_path: Path) -> None:
    output = tmp_path / "profile.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_sixdof_reset_profile.py"),
            "--reset-profile",
            "obstacle_close_live",
            "--num-envs",
            "128",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text())
    assert report["reset_profile"] == "obstacle_close_live"
    assert report["reset_ranges"]["hmin"]["p10"] < 0.35
    assert output.with_suffix(".md").exists()
