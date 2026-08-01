from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import torch

from flightrl.sixdof import build_checkpoint_payload
from flightrl.sixdof.rl import SixDofActorCritic


ROOT = Path(__file__).resolve().parents[1]


def test_train_sixdof_ppo_cli_smoke(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ppo.pt"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_sixdof_ppo.py"),
            "--checkpoint",
            str(checkpoint),
            "--updates",
            "1",
            "--num-envs",
            "8",
            "--horizon",
            "4",
            "--hidden-size",
            "16",
            "--minibatch-size",
            "8",
            "--update-epochs",
            "1",
            "--eval-steps",
            "4",
            "--eval-num-envs",
            "4",
            "--imitation-coef",
            "0.1",
            "--reference-coef",
            "0.2",
            "--reward-mode",
            "progress_clearance",
            "--observation-mode",
            "history1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    saved = torch.load(checkpoint, map_location="cpu")
    assert saved["trainer"] == "ppo"
    assert saved["imitation_coef"] == 0.1
    assert saved["reference_coef"] == 0.2
    assert saved["reward_mode"] == "progress_clearance"
    assert saved["observation_mode"] == "history1"
    assert saved["observation_dim"] == 60
    assert checkpoint.with_suffix(".report.json").exists()


def test_train_sixdof_ppo_cli_supports_task_conditioned_init(tmp_path: Path) -> None:
    initial = tmp_path / "initial.pt"
    torch.save(
        build_checkpoint_payload(
            state_dict=SixDofActorCritic(input_dim=30, hidden_size=16).actor.state_dict(),
            tasks=("position_yaw", "obstacle_avoidance"),
            hidden_size=16,
        ),
        initial,
    )
    checkpoint = tmp_path / "ppo_multitask.pt"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_sixdof_ppo.py"),
            "--init-checkpoint",
            str(initial),
            "--checkpoint",
            str(checkpoint),
            "--task",
            "position_yaw",
            "--updates",
            "1",
            "--num-envs",
            "8",
            "--horizon",
            "4",
            "--minibatch-size",
            "8",
            "--update-epochs",
            "1",
            "--eval-steps",
            "4",
            "--eval-num-envs",
            "4",
            "--task-probability",
            "position_yaw=3.0",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    saved = torch.load(checkpoint, map_location="cpu")
    assert saved["task_conditioned"] is True
    assert saved["tasks"] == ["position_yaw", "obstacle_avoidance"]
    assert saved["observation_dim"] == 30
    assert saved["task_sampling_probabilities"]["position_yaw"] == 0.75


def test_train_sixdof_ppo_cli_supports_teacher_residual_controller(tmp_path: Path) -> None:
    initial = tmp_path / "residual.pt"
    actor = SixDofActorCritic(input_dim=28, hidden_size=16).actor
    for parameter in actor.parameters():
        parameter.data.zero_()
    torch.save(
        build_checkpoint_payload(
            state_dict=actor.state_dict(),
            tasks=("circle",),
            hidden_size=16,
            controller="teacher_residual",
            residual_scale=0.1,
        ),
        initial,
    )
    checkpoint = tmp_path / "residual_ppo.pt"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_sixdof_ppo.py"),
            "--init-checkpoint",
            str(initial),
            "--checkpoint",
            str(checkpoint),
            "--task",
            "circle",
            "--updates",
            "1",
            "--num-envs",
            "8",
            "--horizon",
            "4",
            "--minibatch-size",
            "8",
            "--update-epochs",
            "1",
            "--eval-steps",
            "4",
            "--eval-num-envs",
            "4",
            "--controller",
            "teacher_residual",
            "--residual-scale",
            "0.1",
            "--reward-mode",
            "progress_yaw_clearance",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    saved = torch.load(checkpoint, map_location="cpu")
    assert saved["controller"] == "teacher_residual"
    assert saved["residual_scale"] == 0.1
