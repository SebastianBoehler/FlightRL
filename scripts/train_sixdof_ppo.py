from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import torch

from flightrl.sixdof import SixDofCrazyflieEnv, checkpoint_tasks, evaluate_checkpoint_policy, evaluate_policy, gate_status
from flightrl.sixdof.controller import CONTROLLERS
from flightrl.sixdof.dataset import parse_task_probabilities, task_probability_vector
from flightrl.sixdof.observation import OBSERVATION_MODES, observation_dim
from flightrl.sixdof.rl import REWARD_MODES, PpoConfig, SixDofActorCritic, collect_rollout, load_actor_checkpoint, ppo_update
from flightrl.sixdof.tasks import parse_task_spec, task_observation_dim


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a closed-loop PPO-style 6-DoF policy in simulation")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/sixdof_position_yaw_ppo.pt")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--task", default="position_yaw")
    parser.add_argument("--train-tasks", default=None, help="Task spec for task-conditioned PPO. Defaults to init checkpoint tasks when present.")
    parser.add_argument("--task-probability", action="append", default=[], metavar="TASK=WEIGHT", help="Relative rollout sampling weight. Repeatable.")
    parser.add_argument("--reset-profile", default="position_yaw_medium")
    parser.add_argument("--eval-reset-profile", default="position_yaw_medium")
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--action-std", type=float, default=0.25)
    parser.add_argument("--imitation-coef", type=float, default=0.0, help="Teacher-action MSE weight on policy-visited states.")
    parser.add_argument("--reference-coef", type=float, default=0.0, help="MSE weight to keep actor near the initialized policy.")
    parser.add_argument("--reward-mode", default="env", choices=REWARD_MODES)
    parser.add_argument("--observation-mode", default=None, choices=OBSERVATION_MODES)
    parser.add_argument("--controller", default=None, choices=CONTROLLERS, help="Execute actor directly or as a residual around the teacher controller.")
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--eval-steps", type=int, default=400)
    parser.add_argument("--eval-num-envs", type=int, default=128)
    parser.add_argument("--max-yaw-error-rad", type=float, default=None)
    parser.add_argument("--max-yaw-p95-error-rad", type=float, default=None)
    parser.add_argument("--seed", type=int, default=919)
    parser.add_argument("--native-step", action="store_true")
    args = parser.parse_args()
    init_checkpoint = torch.load(args.init_checkpoint, map_location="cpu") if args.init_checkpoint else None
    args.policy_tasks = resolve_policy_tasks(args, init_checkpoint)
    args.task_probabilities = task_probability_vector(args.policy_tasks, parse_task_probabilities(args.task_probability))
    args.observation_mode = args.observation_mode or (init_checkpoint or {}).get("observation_mode") or "base"
    args.controller = args.controller or str((init_checkpoint or {}).get("controller", "policy"))
    args.residual_scale = float(args.residual_scale if args.residual_scale is not None else (init_checkpoint or {}).get("residual_scale", 0.0))
    args.hidden_size = args.hidden_size or int((init_checkpoint or {}).get("hidden_size", 128))
    input_dim = observation_dim(28 + task_observation_dim(args.policy_tasks), args.observation_mode)

    torch.manual_seed(args.seed)
    env = SixDofCrazyflieEnv(
        num_envs=args.num_envs,
        seed=args.seed,
        task=args.task,
        use_native_step=args.native_step,
        reset_profile=args.reset_profile,
    )
    config = PpoConfig(
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        action_std=args.action_std,
        imitation_coef=args.imitation_coef,
        reference_coef=args.reference_coef,
    )
    model = SixDofActorCritic(input_dim=input_dim, hidden_size=args.hidden_size)
    if init_checkpoint:
        validate_init_checkpoint(init_checkpoint, input_dim)
        load_actor_checkpoint(model, init_checkpoint)
    reference_actor = frozen_actor(model) if args.reference_coef > 0.0 else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    best = None
    history = []
    start = perf_counter()
    for update in range(1, args.updates + 1):
        rollout = collect_rollout(
            env,
            model,
            horizon=args.horizon,
            action_std=args.action_std,
            reward_mode=args.reward_mode,
            observation_mode=args.observation_mode,
            tasks=args.policy_tasks,
            rng=env.rng,
            task_probabilities=args.task_probabilities,
            controller=args.controller,
            residual_scale=args.residual_scale,
        )
        losses = ppo_update(model, optimizer, rollout, config, reference_actor)
        if update == 1 or update == args.updates or update % max(1, args.updates // 4) == 0:
            metrics = eval_actor(model, args)
            score = score_metrics(metrics)
            candidate = payload(model, args, metrics, score, update)
            if best is None or candidate["selection_score"] > best["selection_score"]:
                best = candidate
            history.append({"update": update, "mean_reward": metrics["mean_reward"], "selection_score": score, **losses})
            print(
                f"update={update} reward={metrics['mean_reward']:.3f} pos_err={metrics['mean_position_error_m']:.3f} "
                f"yaw={metrics.get('mean_yaw_error_rad', 0.0):.3f} completed={metrics['mean_completed_fraction']:.3f} "
                f"survival={metrics['mean_survival_fraction']:.3f}",
                flush=True,
            )
    assert best is not None
    best["history"] = history
    best["elapsed_s"] = perf_counter() - start
    output = Path(args.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best, output)
    output.with_suffix(".report.json").write_text(json.dumps(report(best, args), indent=2, sort_keys=True) + "\n")
    print(f"checkpoint={output}")
    print(f"report={output.with_suffix('.report.json')}")


def eval_actor(model: SixDofActorCritic, args: argparse.Namespace) -> dict:
    model.actor.eval()
    if args.controller == "teacher_residual":
        return evaluate_checkpoint_policy(
            transient_checkpoint(model, args),
            seed=args.seed + 1000,
            steps=args.eval_steps,
            num_envs=args.eval_num_envs,
            use_native_step=args.native_step,
            eval_tasks=parse_task_spec(args.task),
            reset_profile=args.eval_reset_profile,
        )
    return evaluate_policy(
        model.actor,
        args.policy_tasks,
        seed=args.seed + 1000,
        steps=args.eval_steps,
        num_envs=args.eval_num_envs,
        use_native_step=args.native_step,
        eval_tasks=parse_task_spec(args.task),
        reset_profile=args.eval_reset_profile,
        observation_mode=args.observation_mode,
    )


def score_metrics(metrics: dict) -> float:
    return (
        3.0 * metrics["mean_completed_fraction"]
        + metrics["mean_survival_fraction"]
        + metrics["clearance_p01_m"]
        - metrics["mean_position_error_m"]
        - metrics.get("mean_yaw_error_rad", 0.0)
        - metrics.get("yaw_error_p95_rad", 0.0)
        - metrics.get("action_saturation_fraction", 0.0)
    )


def payload(model: SixDofActorCritic, args: argparse.Namespace, metrics: dict, score: float, update: int) -> dict:
    return {
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.actor.state_dict().items()},
        "task": ",".join(args.policy_tasks),
        "tasks": list(args.policy_tasks),
        "task_conditioned": len(args.policy_tasks) > 1,
        "hidden_size": args.hidden_size,
        "observation_dim": observation_dim(28 + task_observation_dim(args.policy_tasks), args.observation_mode),
        "base_observation_dim": 28,
        "observation_mode": args.observation_mode,
        "action_dim": 4,
        "selection_update": update,
        "selection_score": score,
        "metrics": metrics,
        "trainer": "ppo",
        "controller": args.controller,
        "residual_scale": args.residual_scale,
        "reset_profile": args.reset_profile,
        "eval_reset_profile": args.eval_reset_profile,
        "imitation_coef": args.imitation_coef,
        "reference_coef": args.reference_coef,
        "reward_mode": args.reward_mode,
        "task_sampling_probabilities": {task: float(probability) for task, probability in zip(args.policy_tasks, args.task_probabilities, strict=True)},
        "use_native_step": args.native_step,
        "note": "Closed-loop PPO simulation checkpoint; not approved for live hardware.",
    }


def report(checkpoint: dict, args: argparse.Namespace) -> dict:
    gate = gate_status(
        checkpoint["metrics"],
        min_clearance_m=0.08,
        min_completed_fraction=0.90,
        max_position_error_m=1.00,
        max_yaw_error_rad=args.max_yaw_error_rad,
        max_yaw_p95_error_rad=args.max_yaw_p95_error_rad,
    )
    return {
        "checkpoint": args.checkpoint,
        "gate": gate,
        "metrics": checkpoint["metrics"],
        "history": checkpoint["history"],
        "controller": args.controller,
        "residual_scale": args.residual_scale,
        "thresholds": {
            "max_yaw_error_rad": args.max_yaw_error_rad,
            "max_yaw_p95_error_rad": args.max_yaw_p95_error_rad,
        },
    }


def transient_checkpoint(model: SixDofActorCritic, args: argparse.Namespace) -> dict:
    return {
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.actor.state_dict().items()},
        "task": ",".join(args.policy_tasks),
        "tasks": list(args.policy_tasks),
        "hidden_size": args.hidden_size,
        "observation_dim": observation_dim(28 + task_observation_dim(args.policy_tasks), args.observation_mode),
        "observation_mode": args.observation_mode,
        "controller": args.controller,
        "residual_scale": args.residual_scale,
    }


def frozen_actor(model: SixDofActorCritic):
    actor = copy.deepcopy(model.actor)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def validate_init_checkpoint(checkpoint: dict, input_dim: int) -> None:
    if int(checkpoint.get("observation_dim", 28)) != input_dim:
        raise ValueError("init checkpoint observation_dim does not match --observation-mode")


def resolve_policy_tasks(args: argparse.Namespace, checkpoint: dict | None) -> tuple[str, ...]:
    if args.train_tasks:
        return parse_task_spec(args.train_tasks)
    if checkpoint:
        return checkpoint_tasks(checkpoint)
    return parse_task_spec(args.task)


if __name__ == "__main__":
    main()
