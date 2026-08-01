from __future__ import annotations

from copy import deepcopy
from time import perf_counter

import torch

from flightrl.puffer4_door_imitation import (
    bootstrap_door_policy,
    door_teacher_actions,
    freeze_door_grounder,
    initialize_door_observability,
    load_compatible_policy_state,
)
from flightrl.puffer4_door_grounding import (
    GROUNDING_EVALUATION_APPEARANCE_SEED,
    evaluate_door_grounder,
    train_door_grounder,
)
from flightrl.puffer4_door_mujoco_replay import (
    MUJOCO_EVALUATION_ROOM_SEEDS,
    combined_grounder_gate,
    collect_mujoco_grounding_replay,
    evaluate_mujoco_grounder,
)
from flightrl.puffer4_door_reporting import aggregate_training_logs
from flightrl.puffer4_door_observation import (
    DOOR_EVIDENCE_DIM,
    DOOR_PHASE_DIM,
    DOOR_SENSOR_DIM,
)
from flightrl.puffer4_door_policy import DOOR_PIXELS
import flightrl.puffer4_door_training_gates as _door_training_gates


MIN_ACTION_LOGSTD = -3.0
MAX_ACTION_LOGSTD = -0.5
fixed_door_gate = _door_training_gates.fixed_door_gate
fixed_door_teacher_gate = _door_training_gates.fixed_door_teacher_gate

def train_door_policy(
    args: dict,
    torch_pufferl,
    *,
    observability_checkpoint: dict,
    total_timesteps: int,
    bootstrap_updates: int,
    bootstrap_learning_rate: float,
    bootstrap_max_policy_rollin: float,
    log_interval: int,
    initial_policy_state: dict | None = None,
    grounder_updates: int = 0,
    grounder_learning_rate: float = 0.002,
    grounder_eval_batches: int = 32,
    grounder_evaluation_seed: int = 10_011,
):
    _door_training_gates.require_reset_safe_fixed_door_ppo(total_timesteps)
    args["train"]["total_timesteps"] = total_timesteps
    vec = torch_pufferl._C.create_vec(args, torch_pufferl._C.gpu)
    policy = torch_pufferl.load_policy(args, vec)
    initialize_door_observability(policy, observability_checkpoint)
    warmstart = load_compatible_policy_state(policy, initial_policy_state)
    grounder_training = train_door_grounder(
        policy,
        vec,
        args,
        torch_pufferl,
        updates=grounder_updates,
        learning_rate=grounder_learning_rate,
        selection_batches=min(grounder_eval_batches, 16),
        log_interval=32,
    )
    grounder_metrics = evaluate_door_grounder(
        policy,
        args,
        torch_pufferl,
        batches=grounder_eval_batches,
        seed=grounder_evaluation_seed,
        appearance_seed=GROUNDING_EVALUATION_APPEARANCE_SEED,
        agents=min(vec.total_agents, 256),
        visibility_threshold=grounder_training["best_selection_metrics"][
            "visibility_threshold"
        ],
    )
    mujoco_evaluation = collect_mujoco_grounding_replay(
        room_seeds=MUJOCO_EVALUATION_ROOM_SEEDS,
        seed=40_101,
    )
    mujoco_metrics = evaluate_mujoco_grounder(
        policy.encoder.grounder,
        mujoco_evaluation,
        visibility_threshold=grounder_training["best_selection_metrics"][
            "visibility_threshold"
        ],
    )
    grounder_gate = combined_grounder_gate(
        grounder_metrics,
        mujoco_metrics,
    )
    print(
        f"grounder_native={grounder_metrics} "
        f"grounder_mujoco={mujoco_metrics} gate={grounder_gate}",
        flush=True,
    )
    grounder_report = {
        "training": grounder_training,
        "evaluation": {
            "native": grounder_metrics,
            "mujoco": mujoco_metrics,
            "mujoco_room_seeds": list(MUJOCO_EVALUATION_ROOM_SEEDS),
            "mujoco_samples": mujoco_evaluation.sample_count,
        },
        "gate": grounder_gate,
    }
    if not grounder_gate["passed"]:
        failed_state = deepcopy(policy.state_dict())
        vec.close()
        return (
            None,
            [],
            0.0,
            {"warmstart": warmstart, "grounder": grounder_report},
            failed_state,
        )
    freeze_door_grounder(policy)
    with torch.no_grad():
        policy.decoder.decoder_logstd.fill_(-2.5)
    bootstrap = bootstrap_door_policy(
        policy,
        vec,
        torch_pufferl,
        updates=bootstrap_updates,
        learning_rate=bootstrap_learning_rate,
        max_policy_rollin=bootstrap_max_policy_rollin,
    )
    bootstrap["warmstart"] = warmstart
    bootstrap["grounder"] = grounder_report
    bootstrap_state = deepcopy(policy.state_dict())
    trainer = torch_pufferl.PuffeRL(args, vec, policy, verbose=False)
    epochs = (
        0
        if total_timesteps <= 0
        else max(1, total_timesteps // trainer.batch_size)
    )
    history = []
    pending_env = []
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        trainer.rollouts()
        if trainer.env_logs:
            pending_env.append(trainer.env_logs)
        trainer.train()
        with torch.no_grad():
            trainer.policy.decoder.decoder_logstd.clamp_(
                MIN_ACTION_LOGSTD,
                MAX_ACTION_LOGSTD,
            )
        if epoch == 1 or epoch == epochs or epoch % log_interval == 0:
            logs = trainer.log()
            env = aggregate_training_logs(pending_env)
            pending_env.clear()
            history.append(
                {
                    "epoch": epoch,
                    "agent_steps": trainer.global_step,
                    "sps": logs["SPS"],
                    "env": env,
                    "loss": logs["loss"],
                }
            )
            print(
                f"ppo={epoch}/{epochs} steps={trainer.global_step} "
                f"sps={logs['SPS']:.0f} env={env}",
                flush=True,
            )
    return (
        trainer,
        history,
        perf_counter() - started,
        bootstrap,
        bootstrap_state,
    )


@torch.no_grad()
def evaluate_door_policy(
    policy,
    args: dict,
    torch_pufferl,
    *,
    steps: int,
    seed: int,
    camera_mask: bool,
    agents: int | None = None,
) -> dict[str, float]:
    eval_args = deepcopy(args)
    eval_args["env"]["seed"] = seed
    eval_args["env"]["camera_mask"] = float(camera_mask)
    if agents is not None:
        eval_args["vec"]["total_agents"] = agents
    vec = torch_pufferl._C.create_vec(eval_args, torch_pufferl._C.gpu)
    observations = torch_pufferl._cpu_tensor(
        vec.obs_ptr,
        (vec.total_agents, vec.obs_size),
        torch.float32,
    )
    terminals = torch_pufferl._cpu_tensor(
        vec.terminals_ptr,
        (vec.total_agents,),
        torch.float32,
    )
    vec.reset()
    state = policy.initial_state(vec.total_agents, device="cpu")
    forward = []
    yaw = []
    for _ in range(steps):
        distribution, _, state = policy.forward_eval(observations, state)
        actions = distribution.mean.clamp(-1.0, 1.0).contiguous()
        actions[:, 0].clamp_(0.0, 1.0)
        forward.append(actions[:, 0].cpu())
        yaw.append(actions[:, 1].abs().cpu())
        vec.cpu_step(actions.data_ptr())
        alive = (1.0 - terminals).view(1, -1, 1)
        state = tuple(item * alive for item in state)
    metrics = {key: float(value) for key, value in dict(vec.log()).items()}
    denominator = metrics.get("outside_fov_episode_fraction", 0.0)
    metrics["outside_fov_success_rate"] = (
        metrics.get("outside_fov_success_fraction", 0.0) / denominator
        if denominator > 0.0
        else 0.0
    )
    metrics["forward_action_mean"] = float(torch.cat(forward).mean())
    metrics["yaw_action_p95"] = float(torch.quantile(torch.cat(yaw), 0.95))
    vec.close()
    return metrics


@torch.no_grad()
def evaluate_door_teacher(
    args: dict,
    torch_pufferl,
    *,
    steps: int,
    seed: int,
    agents: int | None = None,
) -> dict[str, float]:
    eval_args = deepcopy(args)
    eval_args["env"]["seed"] = seed
    eval_args["env"]["camera_mask"] = 0.0
    if agents is not None:
        eval_args["vec"]["total_agents"] = agents
    vec = torch_pufferl._C.create_vec(eval_args, torch_pufferl._C.gpu)
    observations = torch_pufferl._cpu_tensor(
        vec.obs_ptr,
        (vec.total_agents, vec.obs_size),
        torch.float32,
    )
    vec.reset()
    action_samples = []
    evidence_samples = []
    evidence_offset = 3 * DOOR_PIXELS + DOOR_SENSOR_DIM + DOOR_PHASE_DIM
    for _ in range(steps):
        actions = door_teacher_actions(observations).contiguous()
        action_samples.append(actions.clone())
        evidence_samples.append(
            observations[
                :, evidence_offset : evidence_offset + DOOR_EVIDENCE_DIM
            ].clone()
        )
        vec.cpu_step(actions.data_ptr())
    metrics = {key: float(value) for key, value in dict(vec.log()).items()}
    denominator = metrics.get("outside_fov_episode_fraction", 0.0)
    metrics["outside_fov_success_rate"] = (
        metrics.get("outside_fov_success_fraction", 0.0) / denominator
        if denominator > 0.0
        else 0.0
    )
    metrics["outside_fov_observed_rate"] = (
        metrics.get("outside_fov_observed_fraction", 0.0) / denominator
        if denominator > 0.0
        else 0.0
    )
    actions = torch.cat(action_samples)
    evidence = torch.cat(evidence_samples)
    detected = evidence[:, 0] > 0.0
    metrics["teacher_forward_mean"] = float(actions[:, 0].mean())
    metrics["teacher_forward_fraction"] = float(
        (actions[:, 0] > 0.0).float().mean()
    )
    metrics["detector_visible_fraction"] = float(detected.float().mean())
    if torch.any(detected):
        metrics["detector_abs_x_median"] = float(
            evidence[detected, 1].abs().median()
        )
        metrics["detector_scale_median"] = float(evidence[detected, 3].median())
        metrics["detector_scale_p95"] = float(
            torch.quantile(evidence[detected, 3], 0.95)
        )
    vec.close()
    return metrics
