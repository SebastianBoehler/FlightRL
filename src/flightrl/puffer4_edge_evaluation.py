from __future__ import annotations

from copy import deepcopy
import torch

from flightrl.puffer4_door_teacher import finite_metrics, safe_fraction
from flightrl.puffer4_edge_dataset import EDGE_STUDENT_OBSERVATION_DIM
from flightrl.puffer4_edge_evaluation_gate import (
    collision_rate_upper_95,
    edge_student_gate,
)
from flightrl.puffer4_edge_policy import EdgeNavigationActor
from flightrl.puffer4_edge_schema import EDGE_ACTION_DIM, EDGE_OBSERVATION_DIM
from flightrl.puffer4_edge_training import apply_recurrent_resets


@torch.no_grad()
def evaluate_edge_student(
    args: dict,
    torch_pufferl,
    actor: EdgeNavigationActor,
    *,
    steps: int,
    agents: int,
    seed: int,
    appearance_seed: int,
    profile: dict[str, float],
) -> dict:
    if type(steps) is not int or steps <= 0 or type(agents) is not int or agents <= 0:
        raise ValueError("edge evaluation steps and agents must be positive")
    eval_args = deepcopy(args)
    eval_args["env"].update(
        {"seed": seed, "appearance_seed": appearance_seed, "camera_mask": 0.0}
    )
    eval_args["env"].update(profile)
    eval_args["vec"]["total_agents"] = agents
    vec = torch_pufferl._C.create_vec(eval_args, torch_pufferl._C.gpu)
    totals = _empty_totals()
    try:
        if vec.obs_size != EDGE_STUDENT_OBSERVATION_DIM:
            raise RuntimeError("native edge evaluation observation size is incompatible")
        observations = torch_pufferl._cpu_tensor(
            vec.obs_ptr,
            (agents, vec.obs_size),
            torch.float32,
        )
        terminals = torch_pufferl._cpu_tensor(
            vec.terminals_ptr,
            (agents,),
            torch.float32,
        )
        vec.reset()
        state = actor.initial_state(agents)
        reset = torch.ones(agents, dtype=torch.bool)
        for _ in range(steps):
            state = apply_recurrent_resets(state, reset)
            model_observation, target_action, target_grounding = (
                _evaluation_observation_views(observations)
            )
            action, grounding, state = actor.forward_step(
                model_observation.clone(),
                state,
            )
            _accumulate(
                totals,
                action,
                grounding,
                state,
                target_action,
                target_grounding,
                reset,
            )
            vec.cpu_step(action.contiguous().data_ptr())
            reset = terminals > 0.5
        metrics = finite_metrics(dict(vec.log()))
    finally:
        vec.close()
    metrics["outside_fov_success_rate"] = safe_fraction(
        metrics.get("outside_fov_success_fraction", 0.0),
        metrics.get("outside_fov_episode_fraction", 0.0),
    )
    metrics["episodes"] = metrics.get("n", 0.0)
    metrics["collision_rate_upper_95"] = collision_rate_upper_95(
        metrics.get("collision_rate"),
        metrics["episodes"],
    )
    metrics["outside_fov_episodes"] = (
        metrics["episodes"] * metrics.get("outside_fov_episode_fraction", 0.0)
    )
    metrics.update(_finish_totals(totals, steps * agents))
    return {
        "metrics": metrics,
        "gate": edge_student_gate(metrics, profile=profile),
        "seed": seed,
        "appearance_seed": appearance_seed,
        "profile": dict(profile),
        "steps": steps,
        "agents": agents,
    }


def _evaluation_observation_views(
    observations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    action_end = EDGE_OBSERVATION_DIM + EDGE_ACTION_DIM
    return (
        observations[:, :EDGE_OBSERVATION_DIM],
        observations[:, EDGE_OBSERVATION_DIM:action_end],
        observations[:, action_end:EDGE_STUDENT_OBSERVATION_DIM],
    )


def _empty_totals() -> dict[str, float]:
    return {
        "action_squared_error": 0.0,
        "door_action_squared_error": 0.0,
        "reset_action_squared_error": 0.0,
        "reset_door_action_squared_error": 0.0,
        "reset_action_values": 0.0,
        "lateral_action_abs": 0.0,
        "vertical_action_abs": 0.0,
        "lateral_action_abs_max": 0.0,
        "vertical_action_abs_max": 0.0,
        "action_saturated": 0.0,
        "visible_true_positive": 0.0,
        "visible_false_positive": 0.0,
        "visible_false_negative": 0.0,
        "visible_box_abs_error": 0.0,
        "visible_box_values": 0.0,
        "hidden_min": 6.0,
        "hidden_max": 0.0,
    }


def _accumulate(
    totals,
    action,
    grounding,
    state,
    target_action,
    target_grounding,
    reset,
) -> None:
    squared_error = (action - target_action).square()
    totals["action_squared_error"] += float(squared_error.sum())
    totals["door_action_squared_error"] += float(squared_error[:, (0, 3)].sum())
    if bool(reset.any()):
        totals["reset_action_squared_error"] += float(squared_error[reset].sum())
        totals["reset_door_action_squared_error"] += float(
            squared_error[reset][:, (0, 3)].sum()
        )
        totals["reset_action_values"] += float(reset.sum()) * EDGE_ACTION_DIM
    totals["lateral_action_abs"] += float(action[:, 1].abs().sum())
    totals["vertical_action_abs"] += float(action[:, 2].abs().sum())
    totals["lateral_action_abs_max"] = max(
        totals["lateral_action_abs_max"], float(action[:, 1].abs().max())
    )
    totals["vertical_action_abs_max"] = max(
        totals["vertical_action_abs_max"], float(action[:, 2].abs().max())
    )
    totals["action_saturated"] += float((action.abs() >= 0.98).sum())
    predicted = grounding[:, 0] >= 0.5
    actual = target_grounding[:, 0] > 0.5
    totals["visible_true_positive"] += float((predicted & actual).sum())
    totals["visible_false_positive"] += float((predicted & ~actual).sum())
    totals["visible_false_negative"] += float((~predicted & actual).sum())
    if bool(actual.any()):
        totals["visible_box_abs_error"] += float(
            (grounding[actual, 1:] - target_grounding[actual, 1:]).abs().sum()
        )
        totals["visible_box_values"] += float(actual.sum()) * 3.0
    totals["hidden_min"] = min(totals["hidden_min"], float(state.min()))
    totals["hidden_max"] = max(totals["hidden_max"], float(state.max()))


def _finish_totals(totals: dict[str, float], samples: int) -> dict[str, float]:
    tp = totals["visible_true_positive"]
    fp = totals["visible_false_positive"]
    fn = totals["visible_false_negative"]
    return {
        "action_rmse": (
            totals["action_squared_error"] / (EDGE_ACTION_DIM * samples)
        ) ** 0.5,
        "door_action_rmse": (
            totals["door_action_squared_error"] / (2.0 * samples)
        ) ** 0.5,
        "lateral_action_abs_mean": totals["lateral_action_abs"] / samples,
        "vertical_action_abs_mean": totals["vertical_action_abs"] / samples,
        "lateral_action_abs_max": totals["lateral_action_abs_max"],
        "vertical_action_abs_max": totals["vertical_action_abs_max"],
        "action_saturation_fraction": totals["action_saturated"]
        / (EDGE_ACTION_DIM * samples),
        "grounding_visibility_precision": tp / max(tp + fp, 1.0),
        "grounding_visibility_recall": tp / max(tp + fn, 1.0),
        "grounding_visible_box_mae": totals["visible_box_abs_error"]
        / max(totals["visible_box_values"], 1.0),
        "grounding_visible_samples": tp + fn,
        "grounding_absent_samples": samples - tp - fn,
        "reset_action_rmse": (
            totals["reset_action_squared_error"]
            / max(totals["reset_action_values"], 1.0)
        ) ** 0.5,
        "reset_door_action_rmse": (
            totals["reset_door_action_squared_error"]
            / max(totals["reset_action_values"] / 2.0, 1.0)
        ) ** 0.5,
        "reset_samples": totals["reset_action_values"] / EDGE_ACTION_DIM,
        "hidden_min": totals["hidden_min"],
        "hidden_max": totals["hidden_max"],
    }
