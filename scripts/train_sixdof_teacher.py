from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flightrl.sixdof import SixDofCrazyflieEnv, SixDofPolicy, evaluate_policy, teacher_actions
from flightrl.sixdof.tasks import MULTITASK, TASKS, append_task_encoding, parse_task_spec, select_task_actions, task_observation_dim


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a vectorized 6-DoF Crazyflie teacher-imitation policy")
    parser.add_argument("--task", default="position_yaw", help=f"One of {', '.join(TASKS)}, '{MULTITASK}', or comma-separated tasks")
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
    parser.add_argument("--eval-steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tasks = parse_task_spec(args.task)
    rng = np.random.default_rng(args.seed)
    env = SixDofCrazyflieEnv(num_envs=args.num_envs, seed=args.seed, task=tasks[0], use_native_step=args.native_step)
    input_dim = 28 + task_observation_dim(tasks)
    model = SixDofPolicy(hidden_size=args.hidden_size, input_dim=input_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    obs, _ = env.reset(seed=args.seed)

    best_metric = -float("inf")
    best_payload = None
    for update in range(1, args.updates + 1):
        if args.reset_each_update:
            obs, _ = env.reset(seed=args.seed + update)
        obs_batch: list[np.ndarray] = []
        act_batch: list[np.ndarray] = []
        for _ in range(args.steps_per_update):
            task_indices = sample_task_indices(rng, env.num_envs, tasks)
            labels = teacher_labels(env, tasks, task_indices)
            model_obs = append_task_encoding(obs.copy(), task_indices, len(tasks))
            obs_batch.append(model_obs)
            act_batch.append(labels.copy())
            actions = labels
            if update >= args.student_rollout_after and np.random.random() < args.student_rollout_prob:
                with torch.no_grad():
                    actions = model(torch.from_numpy(model_obs).float()).cpu().numpy()
            obs, _rewards, terminals, truncations, _info = env.step(actions)
            if np.any(terminals) or np.any(truncations):
                obs = env.reset_done(terminals | truncations)

        loss = train_epoch(model, optimizer, np.concatenate(obs_batch), np.concatenate(act_batch), args.batch_size)
        if update == 1 or update % max(1, args.updates // 10) == 0:
            metrics = evaluate_policy(model, tasks, seed=args.seed + update, steps=args.eval_steps, use_native_step=args.native_step)
            score = checkpoint_score(metrics)
            if score > best_metric:
                best_metric = score
                best_payload = checkpoint_payload(model, args, tasks, input_dim, metrics, update, score)
            print(
                f"update={update} loss={loss:.6f} "
                f"reward={metrics['mean_reward']:.3f} pos_err={metrics['mean_position_error_m']:.3f} "
                f"clearance={metrics['clearance_p01_m']:.3f} score={score:.3f}",
                flush=True,
            )

    checkpoint_name = args.task.replace(",", "_")
    checkpoint = Path(args.checkpoint or f"artifacts/checkpoints/sixdof_{checkpoint_name}.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_policy(model, tasks, seed=args.seed + 999, steps=args.eval_steps, use_native_step=args.native_step)
    final_score = checkpoint_score(metrics)
    payload = checkpoint_payload(model, args, tasks, input_dim, metrics, args.updates, final_score)
    if best_payload is not None and best_payload["selection_score"] > payload["selection_score"]:
        payload = best_payload
        payload["selected_from"] = "best_eval"
        payload["final_metrics"] = metrics
        payload["final_selection_score"] = final_score
    torch.save(payload, checkpoint)
    print(f"checkpoint={checkpoint}")
    print(f"metrics={payload['metrics']}")


def sample_task_indices(rng: np.random.Generator, num_envs: int, tasks: tuple[str, ...]) -> np.ndarray:
    if len(tasks) == 1:
        return np.zeros(num_envs, dtype=np.int64)
    return rng.integers(0, len(tasks), size=num_envs, dtype=np.int64)


def teacher_labels(env: SixDofCrazyflieEnv, tasks: tuple[str, ...], task_indices: np.ndarray) -> np.ndarray:
    if len(tasks) == 1:
        return teacher_actions(env, task=tasks[0])
    return select_task_actions({task: teacher_actions(env, task=task) for task in tasks}, task_indices, tasks)


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


def checkpoint_score(metrics: dict) -> float:
    return (
        metrics["mean_reward"]
        - metrics["mean_position_error_m"]
        + 0.5 * metrics["mean_completed_fraction"]
        + 0.25 * metrics["clearance_p01_m"]
    )


def checkpoint_payload(model, args, tasks: tuple[str, ...], input_dim: int, metrics: dict, update: int, score: float) -> dict:
    return {
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "task": args.task,
        "tasks": list(tasks),
        "task_conditioned": len(tasks) > 1,
        "hidden_size": args.hidden_size,
        "observation_dim": input_dim,
        "base_observation_dim": 28,
        "action_dim": 4,
        "metrics": metrics,
        "selection_update": update,
        "selection_score": score,
        "use_native_step": args.native_step,
        "note": "Simulation-only 6-DoF teacher imitation checkpoint; not approved for live hardware.",
    }


if __name__ == "__main__":
    main()
