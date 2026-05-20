from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flightrl.sixdof.dataset import load_dataset
from flightrl.sixdof.offline import OfflineTrainConfig, train_offline_policy


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
    parser.add_argument("--eval-num-envs", type=int, default=128)
    parser.add_argument("--select-by-eval", action="store_true")
    parser.add_argument("--native-step", action="store_true")
    parser.add_argument("--eval-reset-profile", default=None, help="Named reset profile used for eval-based selection.")
    parser.add_argument("--action-weighting", default="none", choices=("none", "inverse_std"))
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    config = OfflineTrainConfig(
        dataset=args.dataset,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_ratio=args.val_ratio,
        seed=args.seed,
        eval_steps=args.eval_steps,
        eval_num_envs=args.eval_num_envs,
        select_by_eval=args.select_by_eval,
        use_native_step=args.native_step,
        eval_reset_profile=args.eval_reset_profile,
        action_weighting=args.action_weighting,
    )
    best = train_offline_policy(data, config)
    for entry in best["history"]:
        epoch = entry["epoch"]
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            print(f"epoch={epoch} train_loss={entry['train_loss']:.6f} val_loss={entry['val_loss']:.6f}", flush=True)
    output = Path(args.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best, output)
    print(f"checkpoint={output}")
    print(f"val_loss={best['val_loss']:.6f}")
    print(f"selection_mode={best['selection_mode']}")


if __name__ == "__main__":
    main()
