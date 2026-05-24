from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.env import euler_to_quat
from flightrl.sixdof.policies import teacher_actions
from flightrl.sixdof.rl import PpoConfig, SixDofActorCritic, collect_rollout, compute_advantages, position_error_for_task_indices, ppo_update, rollout_reward


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


def test_collect_rollout_labels_teacher_on_recorded_state() -> None:
    env = SixDofCrazyflieEnv(num_envs=4, seed=5, reset_profile="position_yaw_easy")
    expected = teacher_actions(env, task=env.task)
    rollout = collect_rollout(env, SixDofActorCritic(input_dim=28, hidden_size=16), horizon=1, action_std=0.2)
    np.testing.assert_allclose(rollout["teacher_actions"][0], expected, rtol=1e-6, atol=1e-6)


def test_collect_rollout_teacher_residual_executes_teacher_and_trains_zero_residual() -> None:
    env = SixDofCrazyflieEnv(num_envs=4, seed=6, reset_profile="position_yaw_easy")
    expected = teacher_actions(env, task=env.task)
    rollout = collect_rollout(
        env,
        SixDofActorCritic(input_dim=28, hidden_size=16),
        horizon=1,
        action_std=0.2,
        controller="teacher_residual",
        residual_scale=0.0,
    )

    np.testing.assert_allclose(rollout["executed_actions"][0], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rollout["teacher_actions"][0], np.zeros_like(expected), rtol=1e-6, atol=1e-6)


def test_collect_rollout_supports_progress_reward() -> None:
    env_progress = SixDofCrazyflieEnv(num_envs=4, seed=7, reset_profile="position_yaw_easy")
    env_clearance = SixDofCrazyflieEnv(num_envs=4, seed=7, reset_profile="position_yaw_easy")
    env_base = SixDofCrazyflieEnv(num_envs=4, seed=7, reset_profile="position_yaw_easy")
    model = SixDofActorCritic(input_dim=28, hidden_size=16)
    shaped = collect_rollout(env_progress, model, horizon=2, action_std=0.2, reward_mode="progress")
    clearance = collect_rollout(env_clearance, model, horizon=2, action_std=0.2, reward_mode="progress_clearance")
    raw = collect_rollout(env_base, model, horizon=2, action_std=0.2, reward_mode="env")
    assert shaped["rewards"].shape == raw["rewards"].shape
    assert not np.allclose(shaped["rewards"], raw["rewards"])
    assert not np.allclose(clearance["rewards"], shaped["rewards"])


def test_yaw_clearance_reward_penalizes_true_angle_error() -> None:
    env = SixDofCrazyflieEnv(num_envs=2, seed=9, reset_profile="position_yaw_easy")
    env.quaternion[:] = euler_to_quat(np.zeros(2), np.zeros(2), np.asarray([0.0, np.pi], dtype=np.float32))
    env.target_yaw[:] = 0.0
    env.velocity[:] = 0.0
    env._update_ranges()
    previous_error = np.linalg.norm(env.target_position - env.position, axis=1).astype(np.float32)
    actions = np.zeros((2, 4), dtype=np.float32)
    done = np.zeros(2, dtype=bool)

    clearance = rollout_reward(env, np.zeros(2, dtype=np.float32), done, previous_error, actions, "progress_clearance")
    yaw_clearance = rollout_reward(env, np.zeros(2, dtype=np.float32), done, previous_error, actions, "progress_yaw_clearance")

    assert yaw_clearance[0] == clearance[0]
    assert yaw_clearance[1] < clearance[1] - 1.0


def test_circle_progress_reward_uses_orbit_error_not_center_error() -> None:
    env = SixDofCrazyflieEnv(num_envs=1, seed=10, task="circle", reset_profile="circle_recovery")
    env.position[:] = np.asarray([[0.75, 0.0, 0.65]], dtype=np.float32)
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.65]], dtype=np.float32)

    error = position_error_for_task_indices(env, ("circle",), np.zeros(1, dtype=np.int64))

    assert error[0] < 1e-5
    assert np.linalg.norm(env.target_position - env.position, axis=1)[0] > 0.7


def test_collect_rollout_supports_history_observation_mode() -> None:
    env = SixDofCrazyflieEnv(num_envs=4, seed=11, reset_profile="position_yaw_easy")
    model = SixDofActorCritic(input_dim=60, hidden_size=16)
    rollout = collect_rollout(env, model, horizon=3, action_std=0.2, observation_mode="history1")

    assert rollout["observations"].shape == (3, 4, 60)
    assert rollout["teacher_actions"].shape == rollout["actions"].shape


def test_collect_rollout_supports_task_conditioned_observations() -> None:
    env = SixDofCrazyflieEnv(num_envs=6, seed=12, reset_profile="position_yaw_easy")
    model = SixDofActorCritic(input_dim=30, hidden_size=16)
    rollout = collect_rollout(env, model, horizon=3, action_std=0.2, tasks=("position_yaw", "obstacle_avoidance"), rng=np.random.default_rng(123))

    assert rollout["observations"].shape == (3, 6, 30)
    task_bits = rollout["observations"][:, :, -2:]
    np.testing.assert_allclose(np.sum(task_bits, axis=2), 1.0)
    assert np.any(task_bits[:, :, 0] == 1.0)
    assert np.any(task_bits[:, :, 1] == 1.0)


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
        {
            "state_dict": SixDofActorCritic(input_dim=30, hidden_size=16).actor.state_dict(),
            "hidden_size": 16,
            "observation_dim": 30,
            "observation_mode": "base",
            "tasks": ["position_yaw", "obstacle_avoidance"],
            "task_conditioned": True,
        },
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
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_sixdof_residual_checkpoint.py"),
            "--checkpoint",
            str(initial),
            "--task",
            "circle",
            "--hidden-size",
            "16",
            "--residual-scale",
            "0.1",
            "--zero-weights",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
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
