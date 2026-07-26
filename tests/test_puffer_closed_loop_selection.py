from __future__ import annotations

from pathlib import Path
import importlib.util
from types import SimpleNamespace

from flightrl.sixdof.puffer_policy import PufferPolicyMetadata, PufferSixDofPolicy
from flightrl.sixdof.transfer_test import TransferTestConfig


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_puffer_sixdof_closed_loop.py"
SPEC = importlib.util.spec_from_file_location("train_puffer_sixdof_closed_loop", TRAIN_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
train_puffer_sixdof_closed_loop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_puffer_sixdof_closed_loop)


def test_selection_candidate_can_represent_initial_checkpoint(monkeypatch) -> None:
    metric = {
        "mean_completed_fraction": 1.0,
        "mean_survival_fraction": 1.0,
        "clearance_p01_m": 0.2,
        "mean_position_error_m": 0.1,
        "open_space_horizontal_speed_p95_m_s": 0.3,
        "tilt_p95_deg": 5.0,
        "action_saturation_fraction": 0.0,
        "mean_reward": 1.0,
    }
    reports = {"obstacle_hover_live/python": {"status": "ok", "gate": {"passed": True, "failures": []}, "metrics": metric}}
    monkeypatch.setattr(train_puffer_sixdof_closed_loop, "evaluate_selection_reports", lambda *_args, **_kwargs: reports)
    monkeypatch.setattr(train_puffer_sixdof_closed_loop, "crash_replay_selection_metrics", lambda *_args, **_kwargs: {"crash_replay_l2_p95": 0.0})
    monkeypatch.setattr(train_puffer_sixdof_closed_loop, "crash_replay_selection_score", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(train_puffer_sixdof_closed_loop, "transfer_shadow_selection_metrics", lambda *_args, **_kwargs: {"transfer_shadow_failure_count": 0.0})
    monkeypatch.setattr(train_puffer_sixdof_closed_loop, "transfer_shadow_selection_score", lambda *_args, **_kwargs: 0.0)
    policy = PufferSixDofPolicy(PufferPolicyMetadata(observation_dim=28, hidden_size=16, action_dim=4, num_layers=1))

    candidate, entry, selected = train_puffer_sixdof_closed_loop.selection_candidate(policy, args(), [], TransferTestConfig(), None, update=0)

    assert candidate["selection_update"] == 0
    assert entry["update"] == 0
    assert selected is metric
    assert candidate["selection_score"] == entry["selection_score"]


def test_selection_candidate_uses_fixed_eval_seed_across_updates(monkeypatch) -> None:
    metric = {
        "mean_completed_fraction": 1.0,
        "mean_survival_fraction": 1.0,
        "clearance_p01_m": 0.2,
        "mean_position_error_m": 0.1,
        "open_space_horizontal_speed_p95_m_s": 0.3,
        "tilt_p95_deg": 5.0,
        "action_saturation_fraction": 0.0,
        "mean_reward": 1.0,
    }
    reports = {"obstacle_hover_live/python": {"status": "ok", "gate": {"passed": True, "failures": []}, "metrics": metric}}
    seeds = []
    monkeypatch.setattr(train_puffer_sixdof_closed_loop, "evaluate_selection_reports", lambda *_args, **kwargs: seeds.append(kwargs["seed"]) or reports)
    monkeypatch.setattr(train_puffer_sixdof_closed_loop, "crash_replay_selection_metrics", lambda *_args, **_kwargs: {"crash_replay_l2_p95": 0.0})
    monkeypatch.setattr(train_puffer_sixdof_closed_loop, "crash_replay_selection_score", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(train_puffer_sixdof_closed_loop, "transfer_shadow_selection_metrics", lambda *_args, **_kwargs: {"transfer_shadow_failure_count": 0.0})
    monkeypatch.setattr(train_puffer_sixdof_closed_loop, "transfer_shadow_selection_score", lambda *_args, **_kwargs: 0.0)
    policy = PufferSixDofPolicy(PufferPolicyMetadata(observation_dim=28, hidden_size=16, action_dim=4, num_layers=1))

    train_puffer_sixdof_closed_loop.selection_candidate(policy, args(), [], TransferTestConfig(), None, update=0)
    train_puffer_sixdof_closed_loop.selection_candidate(policy, args(), [], TransferTestConfig(), None, update=8)

    assert seeds == [707, 707]


def args() -> SimpleNamespace:
    return SimpleNamespace(
        crash_replay_action_abs_limit=0.8,
        crash_replay_selection_coef=1.0,
        transfer_selection_coef=1.0,
        selection_backend="python",
        task="obstacle_avoidance",
        reset_profile="obstacle_hover_live",
        eval_reset_profile=None,
        sensor_profile=None,
        physics_profile=None,
        domain_randomization=None,
        disturbance_profile=None,
        eval_disturbance_profile=None,
        disturbance_ramp_start_profile=None,
        disturbance_ramp_updates=0,
        crash_replay_dataset=None,
        crash_replay_coef=0.0,
        crash_replay_envelope_coef=0.0,
        transfer_selection_log=[],
        failed_transfer_selection_log=[],
        transfer_replay_coef=0.0,
        transfer_replay_envelope_coef=0.0,
        transfer_replay_action_abs_limit=0.8,
        previous_action_observation_scale=0.25,
        eval_seed=707,
        reward_mode="puffer_drift_recovery",
        teacher_profile="default",
    )
