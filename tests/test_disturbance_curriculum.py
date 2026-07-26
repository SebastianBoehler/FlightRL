from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.disturbance import SixDofDisturbanceProfile
from flightrl.sixdof.disturbance_curriculum import configure_training_disturbance, interpolate_disturbance_profile, ramp_fraction
from flightrl.sixdof.puffer_policy import PufferPolicyMetadata, PufferSixDofPolicy


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_puffer_sixdof_closed_loop.py"


def test_disturbance_ramp_fraction_spans_training_updates() -> None:
    assert ramp_fraction(update=1, total_updates=5) == 0.0
    assert ramp_fraction(update=3, total_updates=5) == 0.5
    assert ramp_fraction(update=5, total_updates=5) == 1.0
    assert ramp_fraction(update=5, total_updates=10, ramp_updates=3) == 1.0


def test_interpolates_disturbance_ranges() -> None:
    start = SixDofDisturbanceProfile("nominal", world_accel_xy_m_s2=(0.4, 0.5), world_accel_z_m_s2=(-0.01, 0.01))
    end = SixDofDisturbanceProfile("stress", world_accel_xy_m_s2=(4.0, 5.0), world_accel_z_m_s2=(-0.2, -0.1))

    profile = interpolate_disturbance_profile(start, end, 0.25)

    assert profile.name == "nominal_to_stress_0.25"
    assert profile.world_accel_xy_m_s2 == pytest.approx((1.3, 1.625))
    assert profile.world_accel_z_m_s2 == pytest.approx((-0.0575, -0.0175))


def test_training_disturbance_configures_env_from_schedule() -> None:
    env = SixDofCrazyflieEnv(num_envs=4, seed=5, reset_profile="obstacle_hover_live")
    args = argparse.Namespace(
        disturbance_profile=SixDofDisturbanceProfile("stress", world_accel_xy_m_s2=(4.0, 4.0)),
        disturbance_ramp_start_profile=SixDofDisturbanceProfile("nominal", world_accel_xy_m_s2=(0.4, 0.4)),
        disturbance_ramp_updates=3,
    )

    first = configure_training_disturbance(env, args, update=1, total_updates=3)
    final = configure_training_disturbance(env, args, update=3, total_updates=3)

    assert first.world_accel_xy_m_s2 == pytest.approx((0.4, 0.4))
    assert final.world_accel_xy_m_s2 == pytest.approx((4.0, 4.0))
    assert env.disturbance_world_accel.shape == (4, 3)


def test_native_step_rejects_disturbance_curriculum_until_supported() -> None:
    env = SixDofCrazyflieEnv(num_envs=2, seed=3, use_native_step=True)
    args = argparse.Namespace(disturbance_profile=SixDofDisturbanceProfile("stress", world_accel_xy_m_s2=(1.0, 1.0)), disturbance_ramp_start_profile=None, disturbance_ramp_updates=0)

    with pytest.raises(ValueError, match="native 6-DoF step"):
        configure_training_disturbance(env, args, update=1, total_updates=1)


def test_closed_loop_report_persists_disturbance_curriculum(tmp_path: Path) -> None:
    init = tmp_path / "init.bin"
    output = tmp_path / "closed_loop.bin"
    torch.save(PufferSixDofPolicy(PufferPolicyMetadata(observation_dim=28, hidden_size=16, action_dim=4, num_layers=1)).state_dict(), init)

    subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--init-checkpoint",
            str(init),
            "--checkpoint",
            str(output),
            "--updates",
            "1",
            "--num-envs",
            "4",
            "--horizon",
            "2",
            "--eval-steps",
            "2",
            "--eval-num-envs",
            "4",
            "--minibatch-size",
            "4",
            "--update-epochs",
            "1",
            "--disturbance-profile",
            "raw_live_drift",
            "--disturbance-ramp-start-profile",
            "raw_live_mild",
            "--disturbance-ramp-updates",
            "1",
            "--teacher-profile",
            "aggressive_open_stress",
            "--no-wandb",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    context = json.loads(output.with_suffix(".report.json").read_text())["selection_context"]["disturbance_curriculum"]
    report = json.loads(output.with_suffix(".report.json").read_text())

    assert context == {"enabled": True, "start_profile": "raw_live_mild", "end_profile": "raw_live_drift", "ramp_updates": 1}
    assert report["selection_context"]["teacher_profile"] == "aggressive_open_stress"
