from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from flightrl.mujoco.semantic_safety_replay import SafetyReplayConfig


def select_semantic_candidate(
    evaluations: dict[str, dict[str, dict[str, float]]],
    *,
    active_exploration: bool,
) -> str:
    def score(name: str):
        full = evaluations[name]["full"]
        if active_exploration:
            return (
                full.get("collision_rate", 1.0) <= 0.02
                and full.get("unsafe_forward_fraction", 1.0) <= 0.02
                and _moving_clearance(full) >= 0.25
                and full.get("clearance_false_safe_fraction", 1.0) <= 0.02
                and full.get("max_lateral_vertical_action", 1.0) <= 1e-3,
                -full.get("collision_rate", 1.0),
                -full.get("unsafe_forward_fraction", 1.0),
                _moving_clearance(full),
                -full.get("clearance_false_safe_fraction", 1.0),
                -full.get("max_lateral_vertical_action", 1.0),
                full.get("target_discovery_rate", 0.0),
                full.get("success_rate", 0.0) - 2.0 * full.get("collision_rate", 0.0),
                full.get("preacquisition_forward_mean", 0.0),
            )
        return (
            full.get("visible_yaw_sign_accuracy", 0.0) >= 0.90
            and full.get("preacquisition_horizontal_p95_m_s", 1.0) <= 0.03,
            full.get("visible_abs_yawrate_p95_deg_s", 60.0) <= 10.0
            and full.get("max_abs_yawrate_deg_s", 60.0) <= 20.0,
            full.get("success_rate", 0.0) - full.get("collision_rate", 0.0),
            -full.get("visible_yaw_mae_deg_s", 100.0),
        )

    return max(evaluations, key=score)


def build_semantic_training_report(
    *,
    checkpoint: Path,
    digest: str,
    run_tag: str,
    selected: str,
    policy,
    seed: int,
    rooms: int,
    eval_rooms: int,
    bootstrap_updates: int,
    imitation,
    puffer_timesteps: int,
    puffer_elapsed_s: float,
    total_elapsed_s: float,
    max_horizontal_speed_m_s: float,
    history: list[dict],
    evaluations: dict[str, dict[str, dict[str, float]]],
    teacher_evaluation: dict[str, float],
    active_exploration: bool,
    room_profile: str,
    eval_room_profile: str,
    clearance_loss_scale: float,
    collision_risk_loss_scale: float,
) -> dict:
    full = evaluations[selected]["full"]
    target_masked = evaluations[selected]["target_map_masked"]
    vision = policy.encoder.layout.vision
    replay_config = SafetyReplayConfig()
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "run_tag": run_tag,
        "selected_stage": selected,
        "trainer": (
            "PufferLib 4 PuffeRL PPO with stateful transition alignment "
            "after simulator expert bootstrap"
        ),
        "policy": (
            f"{vision.width}x{vision.height} gray4 CNN "
            "+ 16x16 spatial map + MinGRU "
            "+ staged action safety projection"
        ),
        "policy_hidden_size": policy.hidden_size,
        "policy_parameters": sum(
            parameter.numel() for parameter in policy.parameters()
        ),
        "safety_model": (
            "dedicated_recurrent_visual"
            if policy.recurrent_visual_safety is not None
            else (
                "navigation_recurrent"
                if policy.recurrent_safety is not None
                else "dedicated_visual"
            )
        ),
        "safety_target": "forward_action_corridor_clearance",
        "observation_contract": {
            "vision": asdict(policy.encoder.layout.vision),
            "spatial_memory": asdict(policy.encoder.layout.spatial_memory),
            "grounding_tail": {
                "proprioception_index_11": "current detection confidence",
                "proprioception_index_12": "current horizontal image error",
            },
        },
        "training_room_seeds": list(range(seed, seed + rooms)),
        "heldout_room_seeds": list(range(seed + 10_000, seed + 10_000 + eval_rooms)),
        "room_profile": room_profile,
        "eval_room_profile": eval_room_profile,
        "bootstrap_updates": bootstrap_updates,
        "bootstrap_final_action_mse": imitation.action_losses[-1],
        "bootstrap_final_visibility_bce": imitation.visibility_losses[-1],
        "bootstrap_final_clearance_loss": imitation.clearance_losses[-1],
        "bootstrap_final_collision_risk_bce": imitation.collision_risk_losses[-1],
        "safety_replay": {
            **asdict(replay_config),
            "updates": imitation.safety_replay_updates,
            "danger_sequences": imitation.replay_danger_sequences,
            "safe_sequences": imitation.replay_safe_sequences,
            "final_clearance_loss": (
                imitation.replay_clearance_losses[-1]
                if imitation.replay_clearance_losses
                else None
            ),
            "final_collision_risk_bce": (
                imitation.replay_collision_risk_losses[-1]
                if imitation.replay_collision_risk_losses
                else None
            ),
        },
        "safety_loss_scales": {
            "clearance": clearance_loss_scale,
            "collision_risk": collision_risk_loss_scale,
        },
        "bootstrap_phase_weights": {
            "search": 4.0,
            "visible_tracking": 3.0,
            "memory_only": 5.0,
        },
        "puffer_timesteps": puffer_timesteps,
        "puffer_elapsed_s": puffer_elapsed_s,
        "total_elapsed_s": total_elapsed_s,
        "max_horizontal_speed_m_s": max_horizontal_speed_m_s,
        "history": history,
        "evaluation": evaluations,
        "teacher_evaluation": teacher_evaluation,
        "teacher_gate_passed": teacher_gate_passed(teacher_evaluation),
        "target_memory_is_causal": full.get("success_rate", 0.0)
        > target_masked.get("success_rate", 0.0),
        "semantic_conditioning_boundary": (
            "the detector resolves text to target evidence; "
            "the actor is not yet independently language-grounded"
        ),
        "action_projection": {
            "camera_centric_forward_only": active_exploration,
            "no_evidence_translation_m_s": (
                "learned_forward_only" if active_exploration else 0.0
            ),
            "search_yawrate_limit_deg_s": 20.0,
            "visible_target_yawrate_limit_deg_s": 8.0,
        },
        "active_exploration": active_exploration,
        "shadow_gate_passed": shadow_gate_passed(
            full,
            target_masked,
            active_exploration=active_exploration,
            teacher=teacher_evaluation,
        ),
        "deployment_status": "shadow-only until the recorded-flight replay gate passes",
    }


def shadow_gate_passed(
    full: dict[str, float],
    target_masked: dict[str, float],
    *,
    active_exploration: bool,
    teacher: dict[str, float] | None = None,
) -> bool:
    if active_exploration:
        return bool(
            teacher_gate_passed(teacher or {})
            and full.get("success_rate", 0.0) >= 0.50
            and full.get("target_discovery_rate", 0.0) >= 0.70
            and full.get("collision_rate", 1.0) <= 0.02
            and full.get("unsafe_forward_fraction", 1.0) <= 0.02
            and _moving_clearance(full) >= 0.25
            and full.get("clearance_false_safe_fraction", 1.0) <= 0.02
            and full.get("max_lateral_vertical_action", 1.0) <= 1e-3
        )
    return bool(
        full.get("success_rate", 0.0) >= 0.25
        and full.get("collision_rate", 1.0) <= 0.10
        and full.get("preacquisition_horizontal_p95_m_s", 1.0) <= 0.03
        and full.get("visible_yaw_sign_accuracy", 0.0) >= 0.90
        and full.get("visible_abs_yawrate_p95_deg_s", 60.0) <= 10.0
        and full.get("max_abs_yawrate_deg_s", 60.0) <= 20.0
        and full.get("success_rate", 0.0) - target_masked.get("success_rate", 0.0)
        >= 0.15
    )


def teacher_gate_passed(metrics: dict[str, float]) -> bool:
    return bool(
        metrics.get("success_rate", 0.0) >= 0.70
        and metrics.get("target_discovery_rate", 0.0) >= 0.75
        and metrics.get("collision_rate", 1.0) <= 0.02
        and metrics.get("unsafe_forward_fraction", 1.0) <= 0.02
        and _moving_clearance(metrics) >= 0.25
        and metrics.get("max_lateral_vertical_action", 1.0) <= 1e-3
    )


def _moving_clearance(metrics: dict[str, float]) -> float:
    return metrics.get(
        "minimum_moving_navigation_clearance_m",
        metrics.get(
            "minimum_moving_horizontal_clearance_m",
            metrics.get("minimum_moving_front_clearance_m", 0.0),
        ),
    )
