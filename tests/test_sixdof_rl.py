from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.rl import PpoConfig, SixDofActorCritic, collect_rollout, compute_advantages, ppo_update


ROOT = Path(__file__).resolve().parents[1]


def test_compute_advantages_shapes_match_rollout() -> None:
    rollout = {
        "rewards": np.ones((3, 2), dtype=np.float32),
        "dones": np.zeros((3, 2), dtype=np.float32),
        "values": np.zeros((3, 2), dtype=np.float32),
        "next_value": np.zeros(2, dtype=np.float32),
    }
    advantages, returns = compute_advantages(rollout, gamma=0.9, gae_lambda=0.95)
    assert advantages.shape == (6,)
    assert returns.shape == (6,)
    assert float(advantages.max()) > 1.0


def test_ppo_update_runs_on_short_rollout() -> None:
    env = SixDofCrazyflieEnv(num_envs=4, seed=3, reset_profile="position_yaw_easy")
    model = SixDofActorCritic(input_dim=28, hidden_size=16)
    rollout = collect_rollout(env, model, horizon=3, action_std=0.2)
    assert rollout["teacher_actions"].shape == rollout["actions"].shape
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    reference = SixDofActorCritic(input_dim=28, hidden_size=16).actor
    stats = ppo_update(model, optimizer, rollout, PpoConfig(hidden_size=16, minibatch_size=4, update_epochs=1, action_std=0.2, imitation_coef=0.1, reference_coef=0.2), reference)
    assert set(stats) == {"policy_loss", "value_loss", "entropy", "imitation_loss", "reference_loss"}


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
    assert checkpoint.with_suffix(".report.json").exists()
