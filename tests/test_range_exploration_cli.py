from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from flightrl.exploration.range_checkpoint import load_range_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def test_range_training_and_evaluation_cli_create_honest_artifacts(tmp_path: Path) -> None:
    checkpoint = tmp_path / "range_candidate.pt"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_range_exploration.py"),
            "--checkpoint",
            str(checkpoint),
            "--updates",
            "1",
            "--num-envs",
            "2",
            "--horizon",
            "2",
            "--eval-horizon",
            "5",
            "--eval-seeds",
            "1",
            "--seed",
            "701",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    _model, training_report = load_range_checkpoint(checkpoint)
    report_path = checkpoint.with_suffix(".report.json")
    report = json.loads(report_path.read_text())
    assert training_report == report["evaluation"]
    assert report["training"] == {
        "seed": 701,
        "updates": 1,
        "num_envs": 2,
        "rollout_horizon": 2,
        "learning_rate": 3e-4,
        "action_std": 0.25,
        "natural_curriculum_base_count": 256,
        "natural_curriculum_steps": 120,
        "obstacle_curriculum_seed": 200701,
        "obstacle_curriculum_updates": 1,
        "frontier_aux_coef": 0.0,
        "shield_aux_coef": 0.10,
        "general_turn_commitment_coef": 0.0,
        "obstacle_turn_commitment_coef": 0.10,
    }
    assert report["environment"] == {
        "maximum_episode_steps": 1200,
        "step_rate_hz": 20,
    }
    assert report["curriculum"]["schema"] == (
        "flightrl.range_exploration.counterfactual_curriculum.v1"
    )
    assert report["curriculum"]["source"] == "mapper_rollout"
    assert report["curriculum"]["selected_frontier_runtime_input"] is False
    assert report["obstacle_curriculum"] == {
        "schema": "flightrl.range_exploration.obstacle_curriculum.v1",
        "seed": 200701,
        "updates": 1,
        "num_envs": 2,
        "horizon": 2,
        "direction_labels_used": False,
        "actor_selects_yaw": True,
    }
    assert report["objective"] == {
        "frontier_direction_auxiliary_weight": 0.0,
        "shield_consistency_weight": 0.10,
        "general_turn_commitment_weight": 0.0,
        "obstacle_turn_commitment_weight": 0.10,
    }
    assert "counterfactual_curriculum_loss" not in report["history"][0]
    assert report["authority"]["flight"] is False

    reevaluation = tmp_path / "reevaluation.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_range_exploration.py"),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(reevaluation),
            "--horizon",
            "5",
            "--seeds",
            "801",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    evaluated = json.loads(reevaluation.read_text())
    assert evaluated["seeds"] == [801]
    assert evaluated["authority"]["shadow"] is False
