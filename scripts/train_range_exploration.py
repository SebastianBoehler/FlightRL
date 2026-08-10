from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import torch

from flightrl.exploration.range_checkpoint import (
    range_training_contract,
    save_range_checkpoint,
)
from flightrl.exploration.range_batch import RangeExplorationBatch
from flightrl.exploration.range_challenge_training import RangeObstacleTrainingBatch
from flightrl.exploration.range_curriculum import (
    collect_range_natural_counterfactual_batch,
    train_range_natural_curriculum,
)
from flightrl.exploration.range_evaluation import evaluate_range_candidate
from flightrl.exploration.range_policy import RangeExplorationActorCritic
from flightrl.exploration.range_ppo import (
    RangePpoConfig,
    collect_range_rollout,
    range_ppo_update,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the range-frontier exploration v2 map-memory PPO candidate"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--eval-horizon", type=int, default=1_200)
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--action-std", type=float, default=0.25)
    args = parser.parse_args(argv)
    positive = (args.updates, args.num_envs, args.horizon, args.eval_horizon, args.eval_seeds)
    if any(type(value) is not int or value <= 0 for value in positive):
        parser.error("updates, environment counts, and horizons must be positive")
    if args.checkpoint.exists() or args.checkpoint.with_suffix(".report.json").exists():
        parser.error("checkpoint or report output already exists")
    torch.manual_seed(args.seed)
    env = RangeExplorationBatch(
        num_envs=args.num_envs,
        seed=args.seed,
        maximum_episode_steps=1_200,
        stress=True,
    )
    model = RangeExplorationActorCritic(hidden_size=64)
    curriculum_optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    curriculum_observations, curriculum_targets = (
        collect_range_natural_counterfactual_batch(
            seed=args.seed + 100_000,
            base_count=256,
        )
    )
    curriculum = train_range_natural_curriculum(
        model,
        curriculum_optimizer,
        curriculum_observations,
        curriculum_targets,
        seed=args.seed + 100_000,
        steps=120,
        batch_size=64,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    config = RangePpoConfig(action_std=args.action_std)
    history = []
    for update in range(1, args.updates + 1):
        rollout = collect_range_rollout(
            env,
            model,
            horizon=args.horizon,
            action_std=args.action_std,
            reset_seed=args.seed + 10_000 + update * args.num_envs,
        )
        metrics = range_ppo_update(model, optimizer, rollout, config)
        record = {
            "update": update,
            "mean_reward": float(rollout["rewards"].mean()),
            **metrics,
        }
        history.append(record)
        print(
            f"update={update} reward={record['mean_reward']:.6f} "
            f"policy_loss={record['policy_loss']:.6f}",
            flush=True,
        )
    challenge_seed = args.seed + 200_000
    challenge_updates = max(1, args.updates // 2)
    challenge_env = RangeObstacleTrainingBatch(
        num_envs=args.num_envs,
        seed=challenge_seed,
    )
    challenge_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate
    )
    challenge_config = RangePpoConfig(
        action_std=args.action_std,
        turn_commitment_coef=0.10,
    )
    challenge_history = []
    for update in range(1, challenge_updates + 1):
        rollout = collect_range_rollout(
            challenge_env,
            model,
            horizon=args.horizon,
            action_std=args.action_std,
            reset_seed=challenge_seed + 10_000 + update * args.num_envs,
        )
        metrics = range_ppo_update(
            model, challenge_optimizer, rollout, challenge_config
        )
        challenge_history.append(
            {
                "update": update,
                "mean_reward": float(rollout["rewards"].mean()),
                **metrics,
            }
        )
    seeds = tuple(args.seed + 1_000 + index for index in range(args.eval_seeds))
    evaluation = evaluate_range_candidate(
        model,
        seeds=seeds,
        horizon=args.eval_horizon,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    ).stdout.strip()
    training = range_training_contract(
        seed=args.seed,
        updates=args.updates,
        num_envs=args.num_envs,
        rollout_horizon=args.horizon,
        learning_rate=args.learning_rate,
        action_std=args.action_std,
        frontier_aux_coef=config.frontier_aux_coef,
        shield_aux_coef=config.shield_aux_coef,
        general_turn_commitment_coef=config.turn_commitment_coef,
        obstacle_turn_commitment_coef=challenge_config.turn_commitment_coef,
    )
    save_range_checkpoint(
        args.checkpoint,
        model,
        evaluation,
        training=training,
        source_revision=revision,
    )
    report = {
        "schema": "flightrl.range_exploration.training.v7",
        "checkpoint": str(args.checkpoint.resolve()),
        "source_revision": revision,
        "training": training,
        "environment": {
            "maximum_episode_steps": 1_200,
            "step_rate_hz": 20,
        },
        "history": history,
        "curriculum": curriculum,
        "obstacle_curriculum": {
            "schema": "flightrl.range_exploration.obstacle_curriculum.v1",
            "seed": challenge_seed,
            "updates": challenge_updates,
            "num_envs": args.num_envs,
            "horizon": args.horizon,
            "direction_labels_used": False,
            "actor_selects_yaw": True,
        },
        "obstacle_history": challenge_history,
        "objective": {
            "frontier_direction_auxiliary_weight": config.frontier_aux_coef,
            "shield_consistency_weight": config.shield_aux_coef,
            "general_turn_commitment_weight": config.turn_commitment_coef,
            "obstacle_turn_commitment_weight": challenge_config.turn_commitment_coef,
        },
        "evaluation": evaluation,
        "authority": evaluation["authority"],
    }
    report_path = args.checkpoint.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"checkpoint={args.checkpoint}")
    print(f"report={report_path}")
    print(f"simulation_gate_passed={evaluation['simulation_gate_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
