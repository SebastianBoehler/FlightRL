from __future__ import annotations

from pathlib import Path
import importlib.util
import subprocess
import sys

import numpy as np
import torch

from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.crash_selection import crash_replay_selection_metrics, crash_replay_selection_score
from flightrl.sixdof.puffer_policy import PufferPolicyMetadata, PufferSixDofPolicy, load_puffer_sixdof_policy
from flightrl.sixdof.puffer_ppo import (
    PufferPpoConfig,
    collect_puffer_rollout,
    puffer_crash_replay_mse,
    puffer_ppo_update,
    puffer_rollout_reward,
    puffer_transfer_replay_mse,
    transfer_sign_loss,
)
from flightrl.sixdof.transfer_selection import transfer_shadow_selection_score


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_puffer_sixdof_closed_loop.py"
_SPEC = importlib.util.spec_from_file_location("train_puffer_sixdof_closed_loop", TRAIN_SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
train_puffer_sixdof_closed_loop = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(train_puffer_sixdof_closed_loop)


def tiny_policy() -> PufferSixDofPolicy:
    return PufferSixDofPolicy(PufferPolicyMetadata(observation_dim=28, hidden_size=16, action_dim=4, num_layers=1))


def test_puffer_closed_loop_update_runs() -> None:
    env = SixDofCrazyflieEnv(num_envs=4, seed=21, task="obstacle_avoidance", reset_profile="obstacle_hover_live")
    policy = tiny_policy()
    rollout = collect_puffer_rollout(env, policy, horizon=2, action_std=0.05, reward_mode="puffer_hover_transfer")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

    stats = puffer_ppo_update(
        policy,
        optimizer,
        rollout,
        PufferPpoConfig(minibatch_size=4, update_epochs=1, action_std=0.05, reference_coef=0.0),
    )

    assert set(stats) == {
        "policy_loss",
        "value_loss",
        "entropy",
        "imitation_loss",
        "reference_loss",
        "crash_replay_loss",
        "transfer_replay_loss",
    }


def test_puffer_closed_loop_update_applies_crash_replay_regularizer() -> None:
    env = SixDofCrazyflieEnv(num_envs=4, seed=23, task="obstacle_avoidance", reset_profile="obstacle_hover_live")
    policy = tiny_policy()
    rollout = collect_puffer_rollout(env, policy, horizon=2, action_std=0.05, reward_mode="puffer_hover_transfer")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    crash_replay = {
        "observations": torch.tensor(rollout["observations"].reshape(-1, 28), dtype=torch.float32),
        "target_actions": torch.zeros(8, 4),
    }

    stats = puffer_ppo_update(
        policy,
        optimizer,
        rollout,
        PufferPpoConfig(minibatch_size=4, update_epochs=1, action_std=0.05, reference_coef=0.0, crash_replay_coef=0.5),
        crash_replay=crash_replay,
    )

    assert stats["crash_replay_loss"] >= 0.0


def test_crash_replay_envelope_penalizes_large_outputs() -> None:
    policy = tiny_policy()
    with torch.no_grad():
        policy.decoder.decoder_mean.weight.zero_()
        policy.decoder.decoder_mean.bias.fill_(2.0)
    observations = torch.ones(8, 28)
    replay = {"observations": observations, "target_actions": torch.zeros(8, 4)}

    base = puffer_crash_replay_mse(policy, replay, PufferPpoConfig(crash_replay_coef=1.0))
    envelope = puffer_crash_replay_mse(
        policy,
        replay,
        PufferPpoConfig(crash_replay_coef=1.0, crash_replay_envelope_coef=1.0, crash_replay_action_abs_limit=0.0),
    )

    assert envelope >= base


def test_transfer_replay_envelope_penalizes_large_unclamped_outputs() -> None:
    policy = tiny_policy()
    with torch.no_grad():
        policy.decoder.decoder_mean.weight.zero_()
        policy.decoder.decoder_mean.bias.fill_(2.0)
    replay = {"observations": torch.ones(8, 28), "target_actions": torch.zeros(8, 4)}

    base = puffer_transfer_replay_mse(policy, replay, PufferPpoConfig(transfer_replay_coef=1.0))
    envelope = puffer_transfer_replay_mse(
        policy,
        replay,
        PufferPpoConfig(transfer_replay_coef=1.0, transfer_replay_envelope_coef=1.0, transfer_replay_action_abs_limit=0.0),
    )

    assert envelope >= base


def test_transfer_replay_sign_loss_penalizes_wrong_sign() -> None:
    target = torch.tensor([[0.2, -0.2, 0.1, 0.0]])
    correct = torch.tensor([[0.1, -0.1, 0.05, 1.0]])
    wrong = torch.tensor([[-0.1, 0.1, -0.05, 1.0]])

    assert transfer_sign_loss(wrong, target) > transfer_sign_loss(correct, target)


def test_transfer_replay_vertical_mask_adds_roll_pitch_sign_anchor() -> None:
    policy = tiny_policy()
    with torch.no_grad():
        policy.decoder.decoder_mean.weight.zero_()
        policy.decoder.decoder_mean.bias[:] = torch.tensor([0.0, -0.2, -0.2, 0.0])
    replay = {"observations": torch.ones(8, 28), "target_actions": torch.tensor([[0.0, 0.2, 0.2, 0.0]]).repeat(8, 1)}
    vertical = {**replay, "vertical_mask": torch.ones(8, dtype=torch.bool)}

    assert puffer_transfer_replay_mse(policy, vertical, PufferPpoConfig(transfer_replay_coef=1.0)) > puffer_transfer_replay_mse(
        policy,
        replay,
        PufferPpoConfig(transfer_replay_coef=1.0),
    )


def test_crash_replay_selection_penalizes_saturation() -> None:
    policy = tiny_policy()
    replay = {"observations": torch.ones(8, 28), "target_actions": torch.zeros(8, 4)}
    metrics = crash_replay_selection_metrics(policy, replay, action_abs_limit=0.0)

    assert metrics["crash_replay_saturation_fraction"] >= 0.0
    assert crash_replay_selection_score(metrics, action_abs_limit=0.0) <= 0.0


def test_transfer_shadow_selection_penalizes_failures() -> None:
    score = transfer_shadow_selection_score(
        {
            "transfer_shadow_failure_count": 2.0,
            "transfer_shadow_l2_excess": 0.1,
            "transfer_shadow_action_excess": 0.2,
            "transfer_shadow_sign_gap": 0.3,
        }
    )

    assert score < -6.0


def test_numeric_metrics_drops_nested_transfer_labels() -> None:
    metrics = {"transfer_shadow_failure_count": 1.0, "transfer_shadow_labels": {"vertical": {"failures": []}}}

    assert train_puffer_sixdof_closed_loop.numeric_metrics(metrics) == {"transfer_shadow_failure_count": 1.0}


def test_strict_transfer_reward_penalizes_open_space_speed_more() -> None:
    env = SixDofCrazyflieEnv(num_envs=2, seed=22, task="obstacle_avoidance", reset_profile="obstacle_hover_live")
    env.ranges_m[:, :4] = 1.0
    env.velocity[:] = np.asarray([[0.9, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
    previous_error = np.linalg.norm(env.target_position - env.position, axis=1).astype(np.float32)
    actions = np.zeros((2, 4), dtype=np.float32)
    done = np.zeros(2, dtype=bool)

    normal = puffer_rollout_reward(env, np.zeros(2, dtype=np.float32), done, previous_error, actions, "puffer_hover_transfer", (env.task,), np.zeros(2, dtype=np.int64))
    strict = puffer_rollout_reward(
        env,
        np.zeros(2, dtype=np.float32),
        done,
        previous_error,
        actions,
        "puffer_hover_transfer_strict",
        (env.task,),
        np.zeros(2, dtype=np.int64),
    )

    assert strict[0] < normal[0] - 1.0
    assert strict[1] < normal[1]


def test_closed_loop_script_writes_loadable_checkpoint(tmp_path: Path) -> None:
    init = tmp_path / "init.bin"
    output = tmp_path / "closed_loop.bin"
    torch.save(tiny_policy().state_dict(), init)

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
            "--action-std",
            "0.05",
            "--crash-replay-coef",
            "0.0",
            "--physics-profile",
            "crazyflie_brushless",
            "--eval-reset-profile",
            "obstacle_hover_live",
            "--eval-reset-profile",
            "obstacle_hover_drift_recovery",
            "--no-wandb",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    loaded = load_puffer_sixdof_policy(str(output))
    assert loaded.metadata.observation_dim == 28
    assert output.with_suffix(".report.json").exists()


def test_repeated_eval_profiles_are_collected() -> None:
    args = type("Args", (), {"eval_reset_profile": ["obstacle_hover_live", "obstacle_hover_drift_recovery"], "eval_disturbance_profile": ["nominal.json", "stress.json"], "disturbance_profile": "train.json"})()

    assert train_puffer_sixdof_closed_loop.eval_reset_profiles(args) == ["obstacle_hover_live", "obstacle_hover_drift_recovery"]
    assert train_puffer_sixdof_closed_loop.eval_disturbance_profiles(args) == ["nominal.json", "stress.json"]


def test_eval_disturbance_profiles_default_to_training_profile() -> None:
    args = type("Args", (), {"eval_disturbance_profile": None, "disturbance_profile": "train.json"})()

    assert train_puffer_sixdof_closed_loop.eval_disturbance_profiles(args) == ["train.json"]


def test_prepared_transfer_logs_marks_failed_sources(monkeypatch) -> None:
    monkeypatch.setattr(train_puffer_sixdof_closed_loop, "prepare_transfer_selection", lambda specs, failed_source=False: [(type("Case", (), {"failed_source": failed_source})(), []) for _ in specs])
    args = type("Args", (), {"transfer_selection_log": ["clean:a.csv"], "failed_transfer_selection_log": ["failed:b.csv"]})()
    assert [case.failed_source for case, _rows in train_puffer_sixdof_closed_loop.prepared_transfer_logs(args)] == [False, True]


def test_selection_score_penalizes_failed_transfer_backend() -> None:
    metric = {
        "mean_completed_fraction": 1.0,
        "mean_survival_fraction": 1.0,
        "clearance_p01_m": 0.15,
        "mean_position_error_m": 0.1,
        "open_space_horizontal_speed_p95_m_s": 0.6,
        "tilt_p95_deg": 7.0,
        "action_saturation_fraction": 0.0,
    }
    reports = {
        "python": {"status": "ok", "gate": {"passed": True, "failures": []}, "metrics": metric},
        "mujoco": {"status": "ok", "gate": {"passed": False, "failures": ["open_space_horizontal_speed_p95"]}, "metrics": metric},
    }

    assert train_puffer_sixdof_closed_loop.score_reports(reports) < train_puffer_sixdof_closed_loop.score_metrics(metric)
    assert train_puffer_sixdof_closed_loop.score_reports(reports) <= train_puffer_sixdof_closed_loop.score_metrics(metric) - 12.0


def test_selection_metrics_uses_weakest_profile_backend() -> None:
    strong = {
        "mean_completed_fraction": 1.0,
        "mean_survival_fraction": 1.0,
        "clearance_p01_m": 0.2,
        "mean_position_error_m": 0.1,
        "open_space_horizontal_speed_p95_m_s": 0.3,
        "tilt_p95_deg": 5.0,
        "action_saturation_fraction": 0.0,
    }
    weak = {**strong, "mean_position_error_m": 0.8, "open_space_horizontal_speed_p95_m_s": 1.1}

    selected = train_puffer_sixdof_closed_loop.selection_metrics(
        {
            "obstacle_hover_live/python": {"status": "ok", "metrics": strong},
            "obstacle_hover_drift_recovery/python": {"status": "ok", "metrics": weak},
        }
    )

    assert selected is weak
