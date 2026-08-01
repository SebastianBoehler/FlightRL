from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .controller import validate_controller
from .env import ACTION_DIM, OBSERVATION_DIM
from .observation import OBSERVATION_MODES, observation_dim
from .tasks import parse_task_spec, task_observation_dim
from .validation import (
    require_bool,
    require_choice,
    require_finite_real,
    require_positive_int,
)


CHECKPOINT_SCHEMA = "flightrl.sixdof.checkpoint.v1"
CHECKPOINT_CONTRACT_ID = "flightrl-sixdof-state-ranger-rate-policy-v1"
TORCH_POLICY_FORMAT = "torch_sixdof_policy_state_dict"
PUFFER_POLICY_FORMAT = "pufferlib_sixdof_policy_state_dict"
CHECKPOINT_FORMATS = (TORCH_POLICY_FORMAT, PUFFER_POLICY_FORMAT)


@dataclass(frozen=True, slots=True)
class SixDofCheckpointMetadata:
    checkpoint_format: str
    tasks: tuple[str, ...]
    hidden_size: int
    observation_dim: int
    observation_mode: str
    controller: str
    residual_scale: float


def build_checkpoint_payload(
    *,
    state_dict: Mapping[str, object],
    tasks: Sequence[str],
    hidden_size: object,
    observation_mode: str = "base",
    controller: str = "policy",
    residual_scale: object = 0.0,
    checkpoint_format: str = TORCH_POLICY_FORMAT,
) -> dict:
    task_names = _task_names(tasks)
    mode = require_choice(observation_mode, "6-DoF observation mode", OBSERVATION_MODES)
    payload = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_contract": CHECKPOINT_CONTRACT_ID,
        "checkpoint_format": checkpoint_format,
        "state_dict": state_dict,
        "task": ",".join(task_names),
        "tasks": list(task_names),
        "task_conditioned": len(task_names) > 1,
        "hidden_size": require_positive_int(hidden_size, "checkpoint hidden_size"),
        "observation_dim": observation_dim(
            OBSERVATION_DIM + task_observation_dim(task_names),
            mode,
        ),
        "base_observation_dim": OBSERVATION_DIM,
        "observation_mode": mode,
        "action_dim": ACTION_DIM,
        "controller": validate_controller(controller),
        "residual_scale": require_finite_real(
            residual_scale,
            "checkpoint residual_scale",
            minimum=0.0,
        ),
    }
    require_current_checkpoint(payload, expected_format=checkpoint_format)
    return payload


def require_current_checkpoint(
    checkpoint: object,
    *,
    expected_format: str = TORCH_POLICY_FORMAT,
) -> SixDofCheckpointMetadata:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("six-DoF checkpoint must be a mapping")
    if checkpoint.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            "six-DoF checkpoint schema is missing or incompatible; legacy checkpoints are rejected"
        )
    if checkpoint.get("checkpoint_contract") != CHECKPOINT_CONTRACT_ID:
        raise ValueError(
            "six-DoF checkpoint contract is missing or incompatible; legacy checkpoints are rejected"
        )
    checkpoint_format = require_choice(
        checkpoint.get("checkpoint_format"),
        "six-DoF checkpoint format",
        CHECKPOINT_FORMATS,
    )
    if checkpoint_format != expected_format:
        raise ValueError(
            f"six-DoF checkpoint format {checkpoint_format!r} does not match expected {expected_format!r}"
        )

    tasks = _task_names(checkpoint.get("tasks"))
    if checkpoint.get("task") != ",".join(tasks):
        raise ValueError("six-DoF checkpoint task and tasks fields disagree")
    conditioned = require_bool(
        checkpoint.get("task_conditioned"),
        "checkpoint task_conditioned",
    )
    if conditioned != (len(tasks) > 1):
        raise ValueError("six-DoF checkpoint task_conditioned is inconsistent with tasks")

    hidden_size = require_positive_int(
        checkpoint.get("hidden_size"),
        "checkpoint hidden_size",
    )
    base_dim = require_positive_int(
        checkpoint.get("base_observation_dim"),
        "checkpoint base_observation_dim",
    )
    if base_dim != OBSERVATION_DIM:
        raise ValueError("six-DoF checkpoint base observation contract is incompatible")
    mode = require_choice(
        checkpoint.get("observation_mode"),
        "6-DoF observation mode",
        OBSERVATION_MODES,
    )
    actual_observation_dim = require_positive_int(
        checkpoint.get("observation_dim"),
        "checkpoint observation_dim",
    )
    expected_observation_dim = observation_dim(
        OBSERVATION_DIM + task_observation_dim(tasks),
        mode,
    )
    if actual_observation_dim != expected_observation_dim:
        raise ValueError(
            "six-DoF checkpoint observation_dim does not match its task and observation contracts"
        )
    if require_positive_int(checkpoint.get("action_dim"), "checkpoint action_dim") != ACTION_DIM:
        raise ValueError("six-DoF checkpoint action contract is incompatible")

    controller = validate_controller(checkpoint.get("controller"))
    residual_scale = require_finite_real(
        checkpoint.get("residual_scale"),
        "checkpoint residual_scale",
        minimum=0.0,
    )
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("six-DoF checkpoint state_dict must be a non-empty mapping")
    return SixDofCheckpointMetadata(
        checkpoint_format=checkpoint_format,
        tasks=tasks,
        hidden_size=hidden_size,
        observation_dim=actual_observation_dim,
        observation_mode=mode,
        controller=controller,
        residual_scale=residual_scale,
    )


def require_matching_override(value: object, expected: object, name: str) -> None:
    if value is not None and value != expected:
        raise ValueError(
            f"{name} cannot reinterpret an initialized six-DoF checkpoint; expected {expected!r}"
        )


def _task_names(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("six-DoF checkpoint tasks must be a sequence of task names")
    tasks = tuple(value)
    if not all(isinstance(task, str) for task in tasks):
        raise TypeError("six-DoF checkpoint tasks must contain strings")
    parsed = parse_task_spec(",".join(tasks))
    if parsed != tasks or len(set(tasks)) != len(tasks):
        raise ValueError("six-DoF checkpoint tasks must be unique current task names")
    return tasks
