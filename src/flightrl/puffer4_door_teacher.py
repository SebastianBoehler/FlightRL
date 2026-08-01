from __future__ import annotations

from copy import deepcopy
from math import isfinite

import torch

from flightrl.puffer4_door_contract import (
    ACTION_ORDER,
    PRIVILEGED_TAIL_ORDER,
)


def privileged_teacher_gate(metrics: dict[str, float]) -> dict:
    thresholds = {
        "success_rate": (">=", 0.93),
        "collision_rate": ("<=", 0.02),
        "outside_fov_success_rate": (">=", 0.90),
    }
    checks: dict[str, bool] = {}
    for key, (direction, threshold) in thresholds.items():
        value = metrics.get(key)
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(float(value))
            and 0.0 <= float(value) <= 1.0
        )
        checks[key] = valid and (
            float(value) >= threshold
            if direction == ">="
            else float(value) <= threshold
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
    }


@torch.no_grad()
def evaluate_privileged_door_teacher(
    args: dict,
    torch_pufferl,
    *,
    steps: int,
    seed: int,
    agents: int,
) -> dict[str, float]:
    if type(steps) is not int or steps <= 0:
        raise ValueError("teacher evaluation steps must be a positive integer")
    if type(agents) is not int or agents <= 0:
        raise ValueError("teacher evaluation agents must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("teacher evaluation seed must be a nonnegative integer")
    eval_args = deepcopy(args)
    eval_args["env"]["seed"] = seed
    eval_args["env"]["camera_mask"] = 0.0
    eval_args["vec"]["total_agents"] = agents
    vec = torch_pufferl._C.create_vec(eval_args, torch_pufferl._C.gpu)
    try:
        observations = torch_pufferl._cpu_tensor(
            vec.obs_ptr,
            (vec.total_agents, vec.obs_size),
            torch.float32,
        )
        vec.reset()
        action_samples: list[torch.Tensor] = []
        for _ in range(steps):
            actions = privileged_teacher_actions(observations)
            action_samples.append(actions.clone())
            vec.cpu_step(actions.data_ptr())
        metrics = finite_metrics(dict(vec.log()))
    finally:
        vec.close()
    outside_fov = metrics.get("outside_fov_episode_fraction", 0.0)
    metrics["outside_fov_success_rate"] = safe_fraction(
        metrics.get("outside_fov_success_fraction", 0.0),
        outside_fov,
    )
    actions = torch.cat(action_samples)
    metrics["teacher_forward_mean"] = float(actions[:, 0].mean())
    metrics["teacher_forward_fraction"] = float(
        (actions[:, 0] > 0.0).float().mean()
    )
    return metrics


def privileged_teacher_actions(observations: torch.Tensor) -> torch.Tensor:
    if observations.ndim != 2 or observations.shape[1] < len(
        PRIVILEGED_TAIL_ORDER
    ):
        raise ValueError("privileged teacher observations must be a 2D tensor")
    tail_start = observations.shape[1] - len(PRIVILEGED_TAIL_ORDER)
    actions = observations[
        :, tail_start : tail_start + len(ACTION_ORDER)
    ].clone()
    if not bool(torch.isfinite(actions).all()):
        raise ValueError("privileged teacher emitted non-finite actions")
    if bool(torch.any(actions[:, 0] < 0.0) or torch.any(actions[:, 0] > 1.0)):
        raise ValueError("privileged teacher forward action is outside [0, 1]")
    if bool(torch.any(actions[:, 1] < -1.0) or torch.any(actions[:, 1] > 1.0)):
        raise ValueError("privileged teacher yaw action is outside [-1, 1]")
    return actions.contiguous()


def finite_metrics(values: dict) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"teacher metric {key!r} is not numeric")
        parsed = float(value)
        if not isfinite(parsed):
            raise ValueError(f"teacher metric {key!r} is not finite")
        result[str(key)] = parsed
    return result


def safe_fraction(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    result = numerator / denominator
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("teacher conditional success fraction is invalid")
    return result
