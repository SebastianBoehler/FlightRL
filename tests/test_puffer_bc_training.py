from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from flightrl.sixdof.puffer_policy import PufferPolicyMetadata, PufferSixDofPolicy, load_puffer_sixdof_policy


ROOT = Path(__file__).resolve().parents[1]


def test_puffer_bc_training_accepts_init_checkpoint_and_disturbance(tmp_path: Path) -> None:
    init = tmp_path / "init.bin"
    output = tmp_path / "bc.bin"
    torch.save(PufferSixDofPolicy(PufferPolicyMetadata(28, 16, 4, 1)).state_dict(), init)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_puffer_sixdof_bc.py"),
            "--init-checkpoint",
            str(init),
            "--checkpoint",
            str(output),
            "--collect-steps",
            "2",
            "--num-envs",
            "4",
            "--epochs",
            "1",
            "--minibatch-size",
            "4",
            "--disturbance-profile",
            "raw_live_mild",
            "--teacher-profile",
            "aggressive_open_stress",
            "--target-shaping",
            "precontact_drift_brake",
            "--target-shaping-strength",
            "0.25",
            "--previous-action-observation-scale",
            "0.25",
            "--target-mode",
            "current_pose",
            "--policy-envelope-coef",
            "0.1",
            "--policy-action-abs-limit",
            "0.64",
            "--open-space-neutral-coef",
            "0.2",
            "--open-drift-brake-coef",
            "0.3",
            "--drift-speed-m-s",
            "0.5",
            "--no-wandb",
        ],
        cwd=ROOT,
        check=True,
    )

    assert output.exists()
    assert load_puffer_sixdof_policy(output).metadata.hidden_size == 16
    report = json.loads(output.with_suffix(".report.json").read_text())
    assert report["teacher_profile"] == "aggressive_open_stress"
    assert report["target_shaping"] == "precontact_drift_brake"
    assert report["target_shaping_strength"] == 0.25
    assert report["previous_action_observation_scale"] == 0.25
    assert report["target_mode"] == "current_pose"
    assert report["policy_envelope_coef"] == 0.1
    assert report["policy_action_abs_limit"] == 0.64
    assert report["open_space_neutral_coef"] == 0.2
    assert report["open_drift_brake_coef"] == 0.3
    assert report["drift_speed_m_s"] == 0.5


def test_puffer_bc_training_accepts_transfer_replay_log(tmp_path: Path) -> None:
    init = tmp_path / "init.bin"
    output = tmp_path / "bc_transfer.bin"
    log = tmp_path / "transfer.csv"
    failed_log = tmp_path / "failed_transfer.csv"
    torch.save(PufferSixDofPolicy(PufferPolicyMetadata(28, 16, 4, 1)).state_dict(), init)
    log.write_text(
        "stateEstimate.x,stateEstimate.y,stateEstimate.z,stateEstimate.vx,stateEstimate.vy,stateEstimate.vz,"
        "stabilizer.roll,stabilizer.pitch,stabilizer.yaw,range.front,range.back,range.left,range.right,range.up,range.zrange\n"
        "0,0,0.5,0,0,0,0,0,0,800,900,700,600,1500,500\n"
    )
    failed_log.write_text(
        "stateEstimate.x,stateEstimate.y,stateEstimate.z,stateEstimate.vx,stateEstimate.vy,stateEstimate.vz,"
        "stabilizer.roll,stabilizer.pitch,stabilizer.yaw,range.front,range.back,range.left,range.right,range.up,range.zrange,sys.canfly,sys.isTumbled\n"
        "0,0,0.5,0,0,0,0,0,0,800,900,700,600,1500,500,1,0\n"
        "0,0,0.5,0.7,0,0,0,0,0,800,900,700,600,1500,500,1,0\n"
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_puffer_sixdof_bc.py"),
            "--init-checkpoint",
            str(init),
            "--checkpoint",
            str(output),
            "--collect-steps",
            "2",
            "--num-envs",
            "4",
            "--epochs",
            "1",
            "--minibatch-size",
            "4",
            "--transfer-replay-log",
            f"transfer:{log}",
            "--failed-transfer-replay-log",
            f"failed:{failed_log}",
            "--transfer-replay-coef",
            "0.1",
            "--previous-action-observation-scale",
            "0.25",
            "--target-mode",
            "current_pose",
            "--no-wandb",
        ],
        cwd=ROOT,
        check=True,
    )

    assert output.exists()
    report = json.loads(output.with_suffix(".report.json").read_text())
    assert report["transfer_replay_source_rows"] == 3
    assert report["transfer_replay_samples"] == 2
    assert report["transfer_replay_excluded_source_rows"] == 1
    assert report["previous_action_observation_scale"] == 0.25
    assert report["target_mode"] == "current_pose"
