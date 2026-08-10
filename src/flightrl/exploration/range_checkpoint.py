from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from pathlib import Path

import torch

from flightrl.puffer4_edge_training_state import edge_state_dict_sha256

from .range_contract import RANGE_EXPLORATION_CONTRACT_ID
from .range_evaluation import RANGE_EVALUATION_SCHEMA, derive_range_evaluation_checks
from .range_policy import RangeExplorationActorCritic


RANGE_CHECKPOINT_SCHEMA = "flightrl.range_exploration.checkpoint.v7"
_AUTHORITY = {
    "training": False,
    "shadow": False,
    "deployment": False,
    "flight": False,
}
_FIELDS = {
    "schema",
    "contract_id",
    "actor",
    "hidden_size",
    "parameter_count",
    "state_dict",
    "state_sha256",
    "evaluation",
    "training",
    "source_revision",
    "authority",
}


def range_training_contract(
    *,
    seed: int,
    updates: int,
    num_envs: int,
    rollout_horizon: int,
    learning_rate: float,
    action_std: float,
    frontier_aux_coef: float,
    shield_aux_coef: float,
    general_turn_commitment_coef: float,
    obstacle_turn_commitment_coef: float,
) -> dict[str, int | float]:
    value: dict[str, int | float] = {
        "seed": seed,
        "updates": updates,
        "num_envs": num_envs,
        "rollout_horizon": rollout_horizon,
        "learning_rate": learning_rate,
        "action_std": action_std,
        "natural_curriculum_base_count": 256,
        "natural_curriculum_steps": 120,
        "obstacle_curriculum_seed": seed + 200_000,
        "obstacle_curriculum_updates": max(1, updates // 2),
        "frontier_aux_coef": frontier_aux_coef,
        "shield_aux_coef": shield_aux_coef,
        "general_turn_commitment_coef": general_turn_commitment_coef,
        "obstacle_turn_commitment_coef": obstacle_turn_commitment_coef,
    }
    _require_training(value)
    return value


def save_range_checkpoint(
    path: str | Path,
    model: RangeExplorationActorCritic,
    evaluation: dict[str, object],
    *,
    training: dict[str, int | float],
    source_revision: str,
) -> Path:
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    payload = {
        "schema": RANGE_CHECKPOINT_SCHEMA,
        "contract_id": RANGE_EXPLORATION_CONTRACT_ID,
        "actor": "RangeExplorationActorCritic",
        "hidden_size": model.hidden_size,
        "parameter_count": model.parameter_count,
        "state_dict": state,
        "state_sha256": edge_state_dict_sha256(state),
        "evaluation": evaluation,
        "training": training,
        "source_revision": source_revision,
        "authority": dict(_AUTHORITY),
    }
    _require_payload(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return output


def load_range_checkpoint(
    path: str | Path,
) -> tuple[RangeExplorationActorCritic, dict[str, object]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    _require_payload(payload)
    model = RangeExplorationActorCritic(hidden_size=payload["hidden_size"])
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("range checkpoint state is incompatible") from exc
    if model.parameter_count != payload["parameter_count"]:
        raise ValueError("range checkpoint parameter count is incompatible")
    model.eval()
    return model, payload["evaluation"]


def require_shadow_eligible_range_checkpoint(
    path: str | Path,
) -> tuple[RangeExplorationActorCritic, dict[str, object]]:
    model, evaluation = load_range_checkpoint(path)
    if evaluation["simulation_gate_passed"] is not True:
        raise ValueError("range checkpoint has not passed its simulation gate")
    if evaluation["horizon"] != 1_200 or len(evaluation["seeds"]) < 8:
        raise ValueError("range checkpoint lacks the full held-out evaluation envelope")
    return model, evaluation


def _require_payload(value: object) -> None:
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError("range checkpoint fields are incompatible")
    if (
        value.get("schema") != RANGE_CHECKPOINT_SCHEMA
        or value.get("contract_id") != RANGE_EXPLORATION_CONTRACT_ID
        or value.get("actor") != "RangeExplorationActorCritic"
    ):
        raise ValueError("range checkpoint contract is incompatible")
    if value.get("authority") != _AUTHORITY:
        raise ValueError("range checkpoint authority is incompatible")
    hidden_size = value.get("hidden_size")
    parameter_count = value.get("parameter_count")
    if (
        type(hidden_size) is not int
        or hidden_size <= 0
        or type(parameter_count) is not int
        or parameter_count <= 0
    ):
        raise ValueError("range checkpoint architecture is incompatible")
    state = value.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("range checkpoint state is incompatible")
    if value.get("state_sha256") != edge_state_dict_sha256(state):
        raise ValueError("range checkpoint state digest is incompatible")
    _require_evaluation(value.get("evaluation"))
    _require_training(value.get("training"))
    revision = value.get("source_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("range checkpoint source revision is incompatible")
    try:
        int(revision, 16)
    except ValueError as exc:
        raise ValueError("range checkpoint source revision is incompatible") from exc


def _require_evaluation(value: object) -> None:
    required = {
        "schema",
        "scope",
        "seeds",
        "horizon",
        "modes",
        "baselines",
        "counterfactuals",
        "obstacle_challenge",
        "checks",
        "simulation_gate_passed",
        "actor_observation_contains_truth",
        "actor_receives_selected_frontier",
        "authority",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("range checkpoint evaluation fields are incompatible")
    checks = value.get("checks")
    if (
        value.get("schema") != RANGE_EVALUATION_SCHEMA
        or value.get("scope") != "held_out_closed_loop_range_exploration"
        or value.get("authority") != _AUTHORITY
        or value.get("actor_observation_contains_truth") is not False
        or value.get("actor_receives_selected_frontier") is not False
        or not isinstance(checks, dict)
        or not checks
        or any(type(item) is not bool for item in checks.values())
    ):
        raise ValueError("range checkpoint evaluation contract is incompatible")
    derived = all(checks.values())
    if value.get("simulation_gate_passed") is not derived:
        raise ValueError("range checkpoint evaluation checks are contradictory")
    try:
        recomputed = derive_range_evaluation_checks(
            value["modes"],
            value["baselines"],
            value["counterfactuals"],
            value["obstacle_challenge"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("range checkpoint evaluation metrics are incompatible") from exc
    if checks != recomputed:
        raise ValueError("range checkpoint evaluation metrics do not support its checks")
    seeds = value.get("seeds")
    horizon = value.get("horizon")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int for seed in seeds)
        or type(horizon) is not int
        or horizon <= 0
    ):
        raise ValueError("range checkpoint evaluation envelope is incompatible")
    if derived and (horizon != 1_200 or len(seeds) < 8):
        raise ValueError("range checkpoint passed outside the full evaluation envelope")


def _require_training(value: object) -> None:
    fields = {
        "seed",
        "updates",
        "num_envs",
        "rollout_horizon",
        "learning_rate",
        "action_std",
        "natural_curriculum_base_count",
        "natural_curriculum_steps",
        "obstacle_curriculum_seed",
        "obstacle_curriculum_updates",
        "frontier_aux_coef",
        "shield_aux_coef",
        "general_turn_commitment_coef",
        "obstacle_turn_commitment_coef",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("range checkpoint training config is incompatible")
    seed = value["seed"]
    updates = value["updates"]
    integers = ("num_envs", "rollout_horizon")
    positive_floats = ("learning_rate", "action_std")
    if (
        type(seed) is not int
        or seed < 0
        or type(updates) is not int
        or updates < 0
        or any(type(value[name]) is not int or value[name] <= 0 for name in integers)
        or any(
            type(value[name]) is not float
            or not isfinite(value[name])
            or value[name] <= 0.0
            for name in positive_floats
        )
        or value["natural_curriculum_base_count"] != 256
        or value["natural_curriculum_steps"] != 120
        or value["obstacle_curriculum_seed"] != seed + 200_000
        or value["obstacle_curriculum_updates"] != max(1, updates // 2)
        or value["frontier_aux_coef"] != 0.0
        or value["shield_aux_coef"] != 0.10
        or value["general_turn_commitment_coef"] != 0.0
        or value["obstacle_turn_commitment_coef"] != 0.10
    ):
        raise ValueError("range checkpoint training config is incompatible")
