from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flightrl.sixdof import SixDofCrazyflieEnv, SixDofPolicy, teacher_actions


TASKS = ("position_yaw", "obstacle_avoidance", "attitude", "circle")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a vectorized 6-DoF Crazyflie teacher-imitation policy")
    parser.add_argument("--task", choices=TASKS, default="position_yaw")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--updates", type=int, default=240)
    parser.add_argument("--steps-per-update", type=int, default=48)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--student-rollout-after", type=int, default=12)
    parser.add_argument("--student-rollout-prob", type=float, default=0.5)
    parser.add_argument("--reset-each-update", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-step", action="store_true", help="Use the native C 6-DoF dynamics/raycast step")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = SixDofCrazyflieEnv(num_envs=args.num_envs, seed=args.seed, task=args.task, use_native_step=args.native_step)
    model = SixDofPolicy(hidden_size=args.hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    obs, _ = env.reset(seed=args.seed)

    for update in range(1, args.updates + 1):
        if args.reset_each_update:
            obs, _ = env.reset(seed=args.seed + update)
        obs_batch: list[np.ndarray] = []
        act_batch: list[np.ndarray] = []
        for _ in range(args.steps_per_update):
            labels = teacher_actions(env, task=args.task)
            obs_batch.append(obs.copy())
            act_batch.append(labels.copy())
            actions = labels
            if update >= args.student_rollout_after and np.random.random() < args.student_rollout_prob:
                with torch.no_grad():
                    actions = model(torch.from_numpy(obs).float()).cpu().numpy()
            obs, _rewards, terminals, truncations, _info = env.step(actions)
            if np.any(terminals) or np.any(truncations):
                obs, _ = env.reset()

        loss = train_epoch(model, optimizer, np.concatenate(obs_batch), np.concatenate(act_batch), args.batch_size)
        if update == 1 or update % max(1, args.updates // 10) == 0:
            metrics = evaluate(model, args.task, seed=args.seed + update)
            print(
                f"update={update} loss={loss:.6f} "
                f"reward={metrics['mean_reward']:.3f} pos_err={metrics['mean_position_error_m']:.3f} "
                f"clearance={metrics['min_clearance_m']:.3f}",
                flush=True,
            )

    checkpoint = Path(args.checkpoint or f"artifacts/checkpoints/sixdof_{args.task}.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(model, args.task, seed=args.seed + 999, use_native_step=args.native_step)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "task": args.task,
            "hidden_size": args.hidden_size,
            "observation_dim": 28,
            "action_dim": 4,
            "metrics": metrics,
            "use_native_step": args.native_step,
            "note": "Simulation-only 6-DoF teacher imitation checkpoint; not approved for live hardware.",
        },
        checkpoint,
    )
    print(f"checkpoint={checkpoint}")
    print(f"metrics={metrics}")


def train_epoch(model, optimizer, observations: np.ndarray, targets: np.ndarray, batch_size: int) -> float:
    tensor_obs = torch.from_numpy(observations).float()
    tensor_targets = torch.from_numpy(targets).float()
    permutation = torch.randperm(tensor_obs.shape[0])
    losses = []
    for start in range(0, tensor_obs.shape[0], batch_size):
        indices = permutation[start : start + batch_size]
        pred = model(tensor_obs[indices])
        loss = F.mse_loss(pred, tensor_targets[indices])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def evaluate(model: SixDofPolicy, task: str, seed: int, steps: int = 300, use_native_step: bool = False) -> dict[str, float]:
    env = SixDofCrazyflieEnv(num_envs=128, seed=seed, task=task, use_native_step=use_native_step)
    obs, _ = env.reset(seed=seed)
    rewards = []
    min_clearance = []
    for _ in range(steps):
        with torch.no_grad():
            actions = model(torch.from_numpy(obs).float()).cpu().numpy()
        obs, reward, terminals, truncations, _info = env.step(actions)
        rewards.append(reward)
        min_clearance.append(np.min(env.ranges_m[:, :4], axis=1))
        if np.any(terminals) or np.any(truncations):
            obs, _ = env.reset()
    pos_error = np.linalg.norm(env.target_position - env.position, axis=1)
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_position_error_m": float(np.mean(pos_error)),
        "min_clearance_m": float(np.min(min_clearance)),
    }


if __name__ == "__main__":
    main()
