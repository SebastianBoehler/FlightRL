from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import torch

from flightrl.sixdof import SixDofCrazyflieEnv, evaluate_policy, gate_status
from flightrl.sixdof.rl import PpoConfig, SixDofActorCritic, collect_rollout, load_actor_checkpoint, ppo_update


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a closed-loop PPO-style 6-DoF policy in simulation")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/sixdof_position_yaw_ppo.pt")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--task", default="position_yaw")
    parser.add_argument("--reset-profile", default="position_yaw_medium")
    parser.add_argument("--eval-reset-profile", default="position_yaw_medium")
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--action-std", type=float, default=0.25)
    parser.add_argument("--imitation-coef", type=float, default=0.0, help="Teacher-action MSE weight on policy-visited states.")
    parser.add_argument("--reference-coef", type=float, default=0.0, help="MSE weight to keep actor near the initialized policy.")
    parser.add_argument("--eval-steps", type=int, default=400)
    parser.add_argument("--eval-num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=919)
    parser.add_argument("--native-step", action="store_true")
    args = parser.parse_args()

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
    model = SixDofActorCritic(input_dim=28, hidden_size=args.hidden_size)
    if args.init_checkpoint:
        load_actor_checkpoint(model, torch.load(args.init_checkpoint, map_location="cpu"))
    reference_actor = frozen_actor(model) if args.reference_coef > 0.0 else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    best = None
    history = []
    start = perf_counter()
    for update in range(1, args.updates + 1):
        rollout = collect_rollout(env, model, horizon=args.horizon, action_std=args.action_std)
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
                f"completed={metrics['mean_completed_fraction']:.3f} survival={metrics['mean_survival_fraction']:.3f}",
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
    return evaluate_policy(
        model.actor,
        (args.task,),
        seed=args.seed + 1000,
        steps=args.eval_steps,
        num_envs=args.eval_num_envs,
        use_native_step=args.native_step,
        reset_profile=args.eval_reset_profile,
    )


def score_metrics(metrics: dict) -> float:
    return (
        3.0 * metrics["mean_completed_fraction"]
        + metrics["mean_survival_fraction"]
        + metrics["clearance_p01_m"]
        - metrics["mean_position_error_m"]
    )


def payload(model: SixDofActorCritic, args: argparse.Namespace, metrics: dict, score: float, update: int) -> dict:
    return {
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.actor.state_dict().items()},
        "task": args.task,
        "tasks": [args.task],
        "task_conditioned": False,
        "hidden_size": args.hidden_size,
        "observation_dim": 28,
        "base_observation_dim": 28,
        "action_dim": 4,
        "selection_update": update,
        "selection_score": score,
        "metrics": metrics,
        "trainer": "ppo",
        "reset_profile": args.reset_profile,
        "eval_reset_profile": args.eval_reset_profile,
        "imitation_coef": args.imitation_coef,
        "reference_coef": args.reference_coef,
        "use_native_step": args.native_step,
        "note": "Closed-loop PPO simulation checkpoint; not approved for live hardware.",
    }


def report(checkpoint: dict, args: argparse.Namespace) -> dict:
    gate = gate_status(
        checkpoint["metrics"],
        min_clearance_m=0.08,
        min_completed_fraction=0.90,
        max_position_error_m=1.00,
    )
    return {"checkpoint": args.checkpoint, "gate": gate, "metrics": checkpoint["metrics"], "history": checkpoint["history"]}


def frozen_actor(model: SixDofActorCritic):
    actor = copy.deepcopy(model.actor)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


if __name__ == "__main__":
    main()
