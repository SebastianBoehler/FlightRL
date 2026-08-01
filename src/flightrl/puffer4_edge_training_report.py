from __future__ import annotations

from dataclasses import asdict

from flightrl.puffer4_edge_budget import edge_actor_budget
from flightrl.puffer4_edge_contract import edge_policy_contract_report
from flightrl.puffer4_edge_training_data import (
    CRITICAL_DECISION_BOOST,
    MATERIAL_ACTION_SWITCH,
)
from flightrl.puffer4_edge_training_selection import (
    VISUAL_DEPENDENCE_ABSOLUTE_MARGIN,
    VISUAL_DEPENDENCE_RELATIVE_MARGIN,
)
from flightrl.puffer4_edge_training_state import edge_state_dict_sha256


EDGE_TRAINING_REPORT_SCHEMA = "flightrl.edge_v3.training_report.v3"
EDGE_SELECTION_RULE = (
    "minimum_clean_decision_action_loss_after_previous_action_"
    "constant_grounding_and_visual_dependence_gates"
)
EDGE_LOSS_CONTRACT = {
    "action": "episode_balanced_critical_weighted_mse_[1,0.25,0.25,1]",
    "visibility": "episode_and_class_balanced_binary_cross_entropy_with_logits",
    "box": "episode_balanced_visible_only_smooth_l1_center_x_center_y_scale",
    "recurrent": "chronological_tbptt_state_carried_and_detached_at_boundary",
    "previous_action": "exact_stm32_applied_feedback_without_value_masking",
    "visual_ablation": ("selection_only_cyclic_shift_1_of_current_frame_across_agents"),
}
EDGE_WEIGHTING_CONTRACT = {
    "episode": "equal_mass_per_episode_or_censored_tail",
    "visibility": "equal_positive_and_negative_mass_after_episode_balance",
    "critical_events": ["reset", "visibility_transition", "teacher_action_switch"],
    "critical_decision_boost": CRITICAL_DECISION_BOOST,
    "material_action_switch": MATERIAL_ACTION_SWITCH,
    "feedback_observation": "exact_applied_previous_action_in_train_and_selection",
    "visual_dependence": {
        "absolute_decision_action_loss_increase": VISUAL_DEPENDENCE_ABSOLUTE_MARGIN,
        "relative_decision_action_loss_increase": VISUAL_DEPENDENCE_RELATIVE_MARGIN,
        "required_increase": "max_absolute_or_relative_to_clean",
    },
}


def edge_training_report(
    actor,
    config,
    history: list[dict],
    baselines: dict,
    *,
    status: str,
    selected_record: dict,
) -> dict:
    if status not in {"complete", "rejected"}:
        raise ValueError("edge training report status is invalid")
    metrics = dict(selected_record["selection"])
    ablated_metrics = dict(selected_record["selection_visual_ablation"])
    checks = dict(selected_record["baseline_checks"])
    baseline_gate = {"passed": status == "complete", "checks": checks}
    if status == "rejected":
        baseline_gate["failed_checks"] = [
            name for name, passed in checks.items() if not passed
        ]
    return {
        "schema": EDGE_TRAINING_REPORT_SCHEMA,
        "status": status,
        "selection_rule": EDGE_SELECTION_RULE,
        "best_epoch": selected_record["epoch"],
        "best_selection_loss": metrics["selection_score"],
        "best_selection_metrics": metrics,
        "best_selection_visual_ablation_metrics": ablated_metrics,
        "selected_actor_state_sha256": edge_state_dict_sha256(actor.state_dict()),
        "hidden_size": actor.hidden_size,
        "trained_target_ids": [0],
        "config": asdict(config),
        "loss_contract": dict(EDGE_LOSS_CONTRACT),
        "weighting_contract": dict(EDGE_WEIGHTING_CONTRACT),
        "baselines": baselines,
        "baseline_gate": baseline_gate,
        "policy_contract_sha256": edge_policy_contract_report(
            hidden_size=actor.hidden_size
        )["sha256"],
        "model_budget": edge_actor_budget(actor),
        "history": history,
        "authority": "none",
        "deployment_authority": False,
        "hardware_approved": False,
        "controls_drone": False,
    }
