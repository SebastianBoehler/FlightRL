from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flightrl.sixdof import SixDofPolicy, evaluate_policy
from flightrl.sixdof.dataset import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 6-DoF policy on an offline teacher-rollout dataset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/sixdof_offline.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval-steps", type=int, default=800)
    parser.add_argument("--native-step", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data = load_dataset(args.dataset)
    observations = data["observations"]
    actions = data["actions"]
    metadata = data["metadata"]
    tasks = tuple(metadata["tasks"])
    train_idx, val_idx = split_indices(len(observations), args.val_ratio, args.seed)
    model = SixDofPolicy(hidden_size=args.hidden_size, input_dim=observations.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    best = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, optimizer, observations[train_idx], actions[train_idx], args.batch_size)
        val_loss = dataset_loss(model, observations[val_idx], actions[val_idx], args.batch_size)
        if best is None or val_loss < best["val_loss"]:
            best = payload(model, args, metadata, tasks, val_loss, epoch)
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
    assert best is not None
    model.load_state_dict(best["state_dict"])
    best["metrics"] = evaluate_policy(model, tasks, seed=args.seed + 1000, steps=args.eval_steps, use_native_step=args.native_step)
    output = Path(args.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best, output)
    print(f"checkpoint={output}")
    print(f"val_loss={best['val_loss']:.6f}")


def split_indices(count: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(count)
    val_count = max(1, int(count * val_ratio))
    return indices[val_count:], indices[:val_count]


def train_epoch(model, optimizer, observations: np.ndarray, actions: np.ndarray, batch_size: int) -> float:
    order = torch.randperm(len(observations))
    obs = torch.from_numpy(observations).float()
    target = torch.from_numpy(actions).float()
    losses = []
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        loss = F.mse_loss(model(obs[idx]), target[idx])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def dataset_loss(model, observations: np.ndarray, actions: np.ndarray, batch_size: int) -> float:
    losses = []
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            obs = torch.from_numpy(observations[start : start + batch_size]).float()
            target = torch.from_numpy(actions[start : start + batch_size]).float()
            losses.append(float(F.mse_loss(model(obs), target)))
    return float(np.mean(losses))


def payload(model, args, metadata: dict, tasks: tuple[str, ...], val_loss: float, epoch: int) -> dict:
    return {
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "task": ",".join(tasks),
        "tasks": list(tasks),
        "task_conditioned": len(tasks) > 1,
        "hidden_size": args.hidden_size,
        "observation_dim": int(metadata["observation_dim"]),
        "base_observation_dim": 28,
        "action_dim": 4,
        "dataset": args.dataset,
        "selection_epoch": epoch,
        "val_loss": val_loss,
        "note": "Offline teacher-imitation checkpoint; simulation-only and not approved for live hardware.",
    }


if __name__ == "__main__":
    main()
