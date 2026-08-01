from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from flightrl.sixdof.rl import SixDofActorCritic


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_sixdof_ppo", ROOT / "scripts" / "train_sixdof_ppo.py")
assert SPEC and SPEC.loader
PPO = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PPO
SPEC.loader.exec_module(PPO)


def test_ppo_score_penalizes_yaw_when_completion_matches() -> None:
    aligned = metrics(yaw=0.1, yaw_p95=0.3)
    drifting = metrics(yaw=0.8, yaw_p95=1.8)

    assert PPO.score_metrics(aligned) > PPO.score_metrics(drifting)


def test_finalize_best_uses_holdout_seed_not_selection_seed(monkeypatch) -> None:
    model = SixDofActorCritic(input_dim=28, hidden_size=16)
    selection_metrics = metrics(yaw=0.1, yaw_p95=0.3)
    holdout_metrics = metrics(yaw=0.2, yaw_p95=0.4)
    best = {
        "state_dict": model.actor.state_dict(),
        "metrics": selection_metrics,
    }
    calls = []
    monkeypatch.setattr(
        PPO,
        "eval_actor",
        lambda _model, _args, *, seed: calls.append(seed) or holdout_metrics,
    )

    finalized = PPO.finalize_best(
        model,
        best,
        SimpleNamespace(selection_seed=1_919, evaluation_seed=2_919),
    )

    assert calls == [2_919]
    assert finalized["selection_metrics"] is selection_metrics
    assert finalized["metrics"] is holdout_metrics
    assert finalized["selection_seed"] == 1_919
    assert finalized["evaluation_seed"] == 2_919


def metrics(*, yaw: float, yaw_p95: float) -> dict:
    return {
        "mean_completed_fraction": 1.0,
        "mean_survival_fraction": 1.0,
        "clearance_p01_m": 0.5,
        "mean_position_error_m": 0.6,
        "mean_yaw_error_rad": yaw,
        "yaw_error_p95_rad": yaw_p95,
        "action_saturation_fraction": 0.0,
    }
