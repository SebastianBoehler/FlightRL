from __future__ import annotations

import pytest

from flightrl.exploration.student_checkpoint import (
    load_coverage_checkpoint,
    save_coverage_checkpoint,
)
from flightrl.exploration.student_collection import (
    collect_matched_counterfactual_pair,
)
from flightrl.exploration.student_training import (
    CoverageTrainConfig,
    train_coverage_student,
)
from flightrl.mujoco import is_mujoco_available, is_mujoco_rendering_available


pytestmark = pytest.mark.skipif(
    not is_mujoco_available() or not is_mujoco_rendering_available(),
    reason="MuJoCo rendering is unavailable",
)


def test_real_rendered_pair_student_passes_held_out_camera_gate(tmp_path) -> None:
    train = collect_matched_counterfactual_pair(seed=612, split="train")
    selection = collect_matched_counterfactual_pair(seed=613, split="selection")

    actor, report = train_coverage_student(
        train,
        selection,
        CoverageTrainConfig(
            epochs=80,
            learning_rate=1.0e-2,
            tbptt_steps=1,
            seed=7,
        ),
    )

    assert report["causal_gate"]["passed"] is True
    assert report["selection"]["matched_pair_mode_accuracy"] == 1.0
    assert report["selection_history_permuted"][
        "matched_pair_mode_accuracy"
    ] == 0.0
    assert report["selection"]["decision_action_loss"] < report[
        "telemetry_only_baseline"
    ]["decision_action_loss"]
    assert actor.parameter_count <= 50_000

    checkpoint = save_coverage_checkpoint(tmp_path / "student.pt", actor, report)
    restored, restored_report = load_coverage_checkpoint(checkpoint)

    assert restored_report == report
    assert restored.state_dict().keys() == actor.state_dict().keys()

    rejected = {**report, "status": "rejected"}
    rejected_path = tmp_path / "rejected.pt"
    with pytest.raises(ValueError, match="passed causal gate"):
        save_coverage_checkpoint(rejected_path, actor, rejected)
    assert not rejected_path.exists()

    unsupported = {**report, "generalization_authority": True}
    with pytest.raises(ValueError, match="unsupported authority"):
        save_coverage_checkpoint(tmp_path / "unsupported.pt", actor, unsupported)

    dishonest = {
        **report,
        "causal_gate": {
            "checks": {**report["causal_gate"]["checks"], "persistence": False},
            "passed": True,
        },
    }
    with pytest.raises(ValueError, match="causal check details"):
        save_coverage_checkpoint(tmp_path / "dishonest.pt", actor, dishonest)

    contradictory = {
        **report,
        "selection": {
            **report["selection"],
            "decision_action_loss": 2.0,
            "decision_mode_accuracy": 0.0,
        },
    }
    with pytest.raises(ValueError, match="metrics do not support"):
        save_coverage_checkpoint(tmp_path / "contradictory.pt", actor, contradictory)

    mislabeled_scope = {
        **report,
        "evaluation_scope": "held_out_closed_loop_generalization",
        "closed_loop_evaluated": True,
    }
    with pytest.raises(ValueError, match="report contract"):
        save_coverage_checkpoint(tmp_path / "mislabeled.pt", actor, mislabeled_scope)
