from __future__ import annotations

from copy import deepcopy
from time import perf_counter

import torch

from .puffer4_vision_policy import infer_vision_layout


MIN_ACTION_LOGSTD = -2.0
MAX_ACTION_LOGSTD = 0.0
TEACHER_DISTANCE_THRESHOLD = 0.48
BOOTSTRAP_SEQUENCE_LENGTH = 64


def visual_teacher_actions(observations: torch.Tensor) -> torch.Tensor:
    width, height, privileged_dim = infer_vision_layout(observations.shape[1])
    if privileged_dim != 1:
        raise ValueError("visual bootstrap requires the native privileged label")
    intent_start = 3 * width * height
    active = observations[:, intent_start + 3] > TEACHER_DISTANCE_THRESHOLD
    actions = torch.zeros((observations.shape[0], 4), dtype=torch.float32)
    actions[:, 1] = torch.where(active, observations[:, -1], 0.0)
    return actions


def bootstrap_visual_policy(
    policy,
    vec,
    torch_pufferl,
    updates: int,
) -> dict:
    if updates <= 0:
        return {"updates": 0}
    observations = torch_pufferl._cpu_tensor(
        vec.obs_ptr,
        (vec.total_agents, vec.obs_size),
        torch.float32,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-3)
    vec.reset()
    initial_loss = 0.0
    final_loss = 0.0
    for update in range(updates):
        sequence_observations = []
        sequence_targets = []
        for _ in range(BOOTSTRAP_SEQUENCE_LENGTH):
            targets = visual_teacher_actions(observations)
            sequence_observations.append(observations.clone())
            sequence_targets.append(targets)
            vec.cpu_step(targets.contiguous().data_ptr())
        observation_batch = torch.stack(sequence_observations, dim=1)
        target_batch = torch.stack(sequence_targets, dim=1).reshape(-1, 4)
        distribution, _ = policy(observation_batch)
        loss = torch.nn.functional.smooth_l1_loss(distribution.mean, target_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update == 0:
            initial_loss = float(loss.detach())
        final_loss = float(loss.detach())
    return {
        "updates": updates,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "teacher_distance_threshold": TEACHER_DISTANCE_THRESHOLD,
        "sequence_length": BOOTSTRAP_SEQUENCE_LENGTH,
    }


def train_visual_policy(
    args: dict,
    torch_pufferl,
    total_timesteps: int,
    log_interval: int,
    action_logstd: float,
    reset_action_head: bool,
    bootstrap_updates: int,
):
    args["train"]["total_timesteps"] = total_timesteps
    vec = torch_pufferl._C.create_vec(args, torch_pufferl._C.gpu)
    policy = torch_pufferl.load_policy(args, vec)
    with torch.no_grad():
        policy.decoder.decoder_logstd.fill_(action_logstd)
        if reset_action_head:
            policy.decoder.decoder_mean.weight.zero_()
            policy.decoder.decoder_mean.bias.zero_()
    bootstrap = bootstrap_visual_policy(
        policy,
        vec,
        torch_pufferl,
        bootstrap_updates,
    )
    bootstrap_state = {
        key: value.detach().cpu().clone()
        for key, value in policy.state_dict().items()
    }
    trainer = torch_pufferl.PuffeRL(args, vec, policy, verbose=False)
    epochs = max(1, total_timesteps // trainer.batch_size)
    history = []
    pending_env = []
    pending_rollout_s = 0.0
    pending_update_s = 0.0
    pending_epochs = 0
    best_score = float("-inf")
    best_epoch = 0
    best_state = None
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        rollout_started = perf_counter()
        trainer.rollouts()
        pending_rollout_s += perf_counter() - rollout_started
        if trainer.env_logs:
            pending_env.append(trainer.env_logs)
        update_started = perf_counter()
        trainer.train()
        with torch.no_grad():
            trainer.policy.decoder.decoder_logstd.clamp_(
                MIN_ACTION_LOGSTD,
                MAX_ACTION_LOGSTD,
            )
        pending_update_s += perf_counter() - update_started
        pending_epochs += 1
        if epoch == 1 or epoch == epochs or epoch % log_interval == 0:
            logs = trainer.log()
            logs["env"] = _aggregate_episode_logs(pending_env)
            pending_env.clear()
            transitions = pending_epochs * trainer.batch_size
            throughput = {
                "rollout_sps": transitions / pending_rollout_s,
                "update_sps": transitions / pending_update_s,
                "optimizer_sample_sps": (
                    transitions * trainer.config["replay_ratio"] / pending_update_s
                ),
            }
            pending_rollout_s = 0.0
            pending_update_s = 0.0
            pending_epochs = 0
            history.append(
                {
                    "epoch": epoch,
                    "agent_steps": trainer.global_step,
                    "sps": logs["SPS"],
                    "throughput": throughput,
                    "env": logs["env"],
                    "loss": logs["loss"],
                }
            )
            candidate_score = _score_episode_logs(logs["env"])
            if candidate_score > best_score:
                best_score = candidate_score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in trainer.policy.state_dict().items()
                }
            print(
                f"epoch={epoch}/{epochs} steps={trainer.global_step} "
                f"sps={logs['SPS']:.0f} rollout={throughput['rollout_sps']:.0f} "
                f"update={throughput['update_sps']:.0f} env={logs['env']}",
                flush=True,
            )
    if best_state is not None:
        trainer.policy.load_state_dict(best_state)
    elapsed = perf_counter() - started
    return (
        trainer,
        history,
        elapsed,
        {"epoch": best_epoch, "score": best_score},
        {"bootstrap": bootstrap, "state": bootstrap_state},
    )


def _aggregate_episode_logs(logs: list[dict]) -> dict:
    episodes = sum(item.get("n", 0.0) for item in logs)
    if episodes == 0:
        return {}
    keys = set().union(*(item.keys() for item in logs)) - {"n"}
    result = {
        key: sum(item.get(key, 0.0) * item.get("n", 0.0) for item in logs) / episodes
        for key in keys
    }
    result["n"] = episodes
    return result


def _score_episode_logs(logs: dict) -> float:
    if logs.get("n", 0.0) < 32.0:
        return float("-inf")
    return (
        3.0 * logs.get("success_rate", 0.0)
        - 2.0 * logs.get("collision_rate", 0.0)
        + 0.01 * logs.get("episode_return", 0.0)
    )


@torch.no_grad()
def evaluate_visual_policy(
    policy,
    args: dict,
    torch_pufferl,
    steps: int,
    vision_mode: str,
    obstacle_probability: float,
    domain_randomization: float | None = None,
    seed: int | None = None,
) -> dict:
    eval_args = deepcopy(args)
    eval_args["env"]["obstacle_probability"] = obstacle_probability
    if domain_randomization is not None:
        eval_args["env"]["domain_randomization"] = domain_randomization
    if seed is not None:
        eval_args["env"]["seed"] = seed
    vec = torch_pufferl._C.create_vec(eval_args, torch_pufferl._C.gpu)
    obs_dtype = torch.float32 if vec.obs_dtype == "FloatTensor" else torch.uint8
    observations = torch_pufferl._cpu_tensor(
        vec.obs_ptr,
        (vec.total_agents, vec.obs_size),
        obs_dtype,
    )
    terminals = torch_pufferl._cpu_tensor(
        vec.terminals_ptr,
        (vec.total_agents,),
        torch.float32,
    )
    vec.reset()
    state = policy.initial_state(vec.total_agents, device="cpu")
    width, height, _privileged_dim = infer_vision_layout(observations.shape[1])
    vision_dim = 3 * width * height
    lateral_actions = []
    for _ in range(steps):
        policy_obs = observations
        if vision_mode == "masked":
            policy_obs = observations.clone()
            policy_obs[:, :vision_dim] = 0.0
        distribution, _, state = policy.forward_eval(policy_obs, state)
        actions = distribution.mean.clamp(-1.0, 1.0).contiguous()
        lateral_actions.append(actions[:, 1].abs().cpu())
        vec.cpu_step(actions.data_ptr())
        alive = (1.0 - terminals).view(1, -1, 1)
        state = tuple(item * alive for item in state)
    metrics = dict(vec.log())
    lateral = torch.cat(lateral_actions)
    residual_speed = (
        float(eval_args["env"]["navigation_residual_scale"])
        * float(eval_args["env"]["max_horizontal_speed_m_s"])
    )
    metrics["lateral_action_mean"] = float(lateral.mean())
    metrics["lateral_action_p95"] = float(torch.quantile(lateral, 0.95))
    metrics["lateral_residual_p95_m_s"] = (
        metrics["lateral_action_p95"] * residual_speed
    )
    vec.close()
    return {key: float(value) for key, value in metrics.items()}


def visual_simulation_gate(evaluation: dict[str, dict[str, float]]) -> dict:
    obstacle = evaluation["obstacle_full_vision"]
    masked = evaluation["obstacle_masked_vision"]
    clear = evaluation["clear_full_vision"]
    nominal = evaluation["nominal_obstacle_full_vision"]
    checks = {
        "randomized_success": obstacle.get("success_rate", 0.0) >= 0.80,
        "randomized_collision": obstacle.get("collision_rate", 1.0) <= 0.05,
        "clear_success": clear.get("success_rate", 0.0) >= 0.90,
        "nominal_success": nominal.get("success_rate", 0.0) >= 0.90,
        "clear_lateral_action": clear.get("lateral_action_p95", 1.0) <= 0.40,
        "camera_dependence": (
            obstacle.get("success_rate", 0.0) - masked.get("success_rate", 0.0)
        )
        >= 0.50,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
    }
