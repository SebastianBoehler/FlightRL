from __future__ import annotations

from copy import deepcopy
from math import isfinite
from time import perf_counter_ns
from typing import Literal

import torch

from flightrl.puffer4_door_eval_stats import (
    episode_evidence,
    marginal_group_evidence,
    performance_report,
    wilson_interval as wilson_interval,
)
from flightrl.puffer4_door_temporal_ablation import (
    DoorTemporalOrderScrambler,
)

RecurrentMode = Literal["carried", "reset_each_step"]


def _tensor_finite(value: object) -> bool:
    return isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())


@torch.no_grad()
def evaluate_promotion_door_policy(
    policy,
    args: dict,
    torch_pufferl,
    *,
    steps: int,
    seed: int,
    camera_mask: bool,
    agents: int | None = None,
    recurrent_mode: RecurrentMode = "carried",
    yaw_abs_limit_normalized: float | None = None,
    temporal_order_seed: int | None = None,
    observation_transform=None,
) -> dict:
    if steps <= 0:
        raise ValueError("promotion evaluation steps must be positive")
    if recurrent_mode not in {"carried", "reset_each_step"}:
        raise ValueError("unknown promotion recurrent mode")
    if yaw_abs_limit_normalized is not None and not (
        0.0 < yaw_abs_limit_normalized <= 1.0
    ):
        raise ValueError("normalized yaw cap must be in (0, 1]")
    if temporal_order_seed is not None and (
        camera_mask
        or recurrent_mode != "carried"
        or yaw_abs_limit_normalized is not None
    ):
        raise ValueError("temporal-order ablation must run in isolation")
    if observation_transform is not None and (
        camera_mask
        or recurrent_mode != "carried"
        or yaw_abs_limit_normalized is not None
        or temporal_order_seed is not None
    ):
        raise ValueError("observation challenge must run in isolation")

    eval_args = deepcopy(args)
    eval_args["env"]["seed"] = seed
    eval_args["env"]["camera_mask"] = float(camera_mask)
    if agents is not None:
        eval_args["vec"]["total_agents"] = agents
    vec = torch_pufferl._C.create_vec(eval_args, torch_pufferl._C.gpu)
    batch_agents = vec.total_agents
    observations = torch_pufferl._cpu_tensor(
        vec.obs_ptr,
        (batch_agents, vec.obs_size),
        torch.float32,
    )
    terminals = torch_pufferl._cpu_tensor(
        vec.terminals_ptr,
        (batch_agents,),
        torch.float32,
    )
    vec.reset()
    temporal_order = (
        DoorTemporalOrderScrambler(
            agent_count=batch_agents,
            seed=temporal_order_seed,
        )
        if temporal_order_seed is not None
        else None
    )
    initial_state = policy.initial_state(batch_agents, device="cpu")
    state = initial_state
    finite = {
        "observations": True,
        "terminals": True,
        "policy_mean": True,
        "value": True,
        "recurrent_state": True,
        "actions": True,
        "metrics": True,
    }
    violation: str | None = None
    policy_ns: list[int] = []
    env_ns: list[int] = []
    loop_ns: list[int] = []
    forward: list[torch.Tensor] = []
    proposed_yaw: list[torch.Tensor] = []
    executed_yaw: list[torch.Tensor] = []
    saturated = 0
    yaw_samples = 0

    try:
        if not _tensor_finite(observations):
            finite["observations"] = False
            violation = "observations"
        for _ in range(steps):
            if violation is not None:
                break
            loop_started = perf_counter_ns()
            input_state = (
                initial_state
                if recurrent_mode == "reset_each_step"
                else state
            )
            policy_observations = observations
            if temporal_order is not None:
                policy_observations = temporal_order.transform(observations)
            elif observation_transform is not None:
                policy_observations = observation_transform.transform(
                    observations
                )
            if not _tensor_finite(policy_observations):
                finite["observations"] = False
                violation = "transformed_observations"
                break
            policy_started = perf_counter_ns()
            distribution, values, next_state = policy.forward_eval(
                policy_observations,
                input_state,
            )
            policy_ns.append(perf_counter_ns() - policy_started)
            means = distribution.mean
            checks = (
                ("policy_mean", _tensor_finite(means)),
                ("value", _tensor_finite(values)),
                (
                    "recurrent_state",
                    bool(next_state)
                    and all(_tensor_finite(item) for item in next_state),
                ),
            )
            failed = next((name for name, passed in checks if not passed), None)
            if failed is not None:
                finite[failed] = False
                violation = failed
                break
            actions = means.clamp(-1.0, 1.0).contiguous()
            actions[:, 0].clamp_(0.0, 1.0)
            proposed = means[:, 1].abs().cpu()
            if yaw_abs_limit_normalized is not None:
                saturated += int(
                    (means[:, 1].abs() > yaw_abs_limit_normalized).sum()
                )
                actions[:, 1].clamp_(
                    -yaw_abs_limit_normalized,
                    yaw_abs_limit_normalized,
                )
            if not _tensor_finite(actions):
                finite["actions"] = False
                violation = "actions"
                break
            forward.append(actions[:, 0].cpu())
            proposed_yaw.append(proposed)
            executed_yaw.append(actions[:, 1].abs().cpu())
            yaw_samples += actions.shape[0]
            env_started = perf_counter_ns()
            vec.cpu_step(actions.data_ptr())
            env_ns.append(perf_counter_ns() - env_started)
            if not _tensor_finite(observations):
                finite["observations"] = False
                violation = "observations"
            elif not _tensor_finite(terminals):
                finite["terminals"] = False
                violation = "terminals"
            loop_ns.append(perf_counter_ns() - loop_started)
            if violation is not None:
                break
            state = next_state
            if recurrent_mode == "carried":
                alive = (1.0 - terminals).view(1, -1, 1)
                state = tuple(item * alive for item in state)
            if temporal_order is not None:
                temporal_order.clear(terminals)
            if observation_transform is not None:
                observation_transform.clear(terminals)

        raw_metrics = {
            key: float(value) for key, value in dict(vec.log()).items()
        }
    finally:
        vec.close()

    invalid_metrics = [
        key for key, value in raw_metrics.items() if not isfinite(value)
    ]
    if invalid_metrics:
        finite["metrics"] = False
        violation = violation or f"metrics:{invalid_metrics[0]}"
    metrics = {
        key: value
        for key, value in raw_metrics.items()
        if isfinite(value)
    }
    outside = metrics.get("outside_fov_episode_fraction", 0.0)
    metrics["outside_fov_success_rate"] = (
        metrics.get("outside_fov_success_fraction", 0.0) / outside
        if outside > 0.0
        else 0.0
    )
    if forward:
        metrics["forward_action_mean"] = float(torch.cat(forward).mean())
        metrics["yaw_proposal_abs_p95"] = float(
            torch.quantile(torch.cat(proposed_yaw), 0.95)
        )
        metrics["yaw_action_p95"] = float(
            torch.quantile(torch.cat(executed_yaw), 0.95)
        )

    completed_steps = len(env_ns)
    finite["passed"] = all(finite.values())
    status = "complete" if violation is None else "aborted_non_finite"
    try:
        counts = episode_evidence(metrics)
    except ValueError as exc:
        counts = {"error": str(exc)}
        if status == "complete":
            status = "invalid_episode_evidence"
    try:
        marginal_groups = marginal_group_evidence(metrics)
    except ValueError as exc:
        marginal_groups = {"status": "invalid", "error": str(exc)}
        if status == "complete":
            status = "invalid_marginal_group_evidence"
    metrics.update(
        {
            "status": status,
            "requested_steps": steps,
            "completed_steps": completed_steps,
            "condition": {
                "camera": "masked" if camera_mask else "full",
                "recurrent_mode": recurrent_mode,
                "temporal_order": (
                    "scrambled" if temporal_order is not None else "ordered"
                ),
                "observation_challenge": observation_transform is not None,
            },
            "finite_outputs": finite | {"first_violation": violation},
            "episode_evidence": counts,
            "marginal_groups": marginal_groups,
            "performance": performance_report(
                batch_agents=batch_agents,
                policy_ns=policy_ns,
                env_ns=env_ns,
                loop_ns=loop_ns,
            ),
            "yaw_cap": {
                "enabled": yaw_abs_limit_normalized is not None,
                "normalized_limit": yaw_abs_limit_normalized,
                "saturation_fraction": (
                    saturated / yaw_samples if yaw_samples else 0.0
                ),
            },
        }
    )
    return metrics


def build_recurrence_reset_ablation(
    carried: dict,
    reset_each_step: dict,
) -> dict:
    keys = ("success_rate", "outside_fov_success_rate", "collision_rate")
    return {
        "label": "recurrent_state_reset_each_step",
        "not_a_temporal_order_shuffle": True,
        "interpretation": (
            "Measures dependence on carried recurrent state; it does not "
            "shuffle or reverse observation order."
        ),
        "condition": {
            "camera": "full",
            "recurrent_mode": "reset_each_step",
        },
        "metrics": reset_each_step,
        "delta_vs_carried": {
            key: reset_each_step.get(key, 0.0) - carried.get(key, 0.0)
            for key in keys
        },
    }
