from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from flightrl.vision import (
    CompactVisionActionPolicy,
    VisionActionPolicyMetadata,
    VisionActionScale,
    load_aligned_vision_actions,
    phase_holdout_split,
    save_vision_action_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact visual trajectory-imitation policy from aligned AI Deck flights.")
    parser.add_argument("--vision", action="append", required=True)
    parser.add_argument("--telemetry", action="append", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--velocity-scale-m-s", type=float, default=0.15)
    parser.add_argument("--yawrate-scale-deg-s", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=818)
    args = parser.parse_args()
    if len(args.vision) != len(args.telemetry):
        raise SystemExit("--vision and --telemetry must be supplied the same number of times")

    start = perf_counter()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    scale = VisionActionScale(args.velocity_scale_m_s, args.yawrate_scale_deg_s)
    dataset = load_aligned_vision_actions(args.vision, args.telemetry, scale=scale)
    train_indices, validation_indices = phase_holdout_split(
        dataset,
        validation_fraction=args.validation_fraction,
    )
    shape = dataset.observations.shape[1:]
    policy = CompactVisionActionPolicy(
        VisionActionPolicyMetadata(
            channels=int(shape[0]),
            height=int(shape[1]),
            width=int(shape[2]),
            hidden_size=args.hidden_size,
            velocity_scale_m_s=args.velocity_scale_m_s,
            yawrate_scale_deg_s=args.yawrate_scale_deg_s,
            contract_json=dataset.contract_json,
        )
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    observations = torch.from_numpy(dataset.observations)
    targets = torch.from_numpy(dataset.actions)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        policy.train()
        permutation = np.random.permutation(train_indices)
        train_loss = 0.0
        for offset in range(0, len(permutation), args.batch_size):
            batch = permutation[offset : offset + args.batch_size]
            prediction = policy(observations[batch])
            loss = F.mse_loss(prediction, targets[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.detach()) * len(batch)
        validation_loss = mse(policy, observations[validation_indices], targets[validation_indices])
        entry = {
            "epoch": epoch,
            "train_mse": train_loss / len(train_indices),
            "validation_mse": validation_loss,
        }
        history.append(entry)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = deepcopy(policy.state_dict())
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch} train_mse={entry['train_mse']:.6f} "
                f"validation_mse={validation_loss:.6f}",
                flush=True,
            )

    assert best_state is not None
    policy.load_state_dict(best_state)
    metrics = action_metrics(policy, observations[validation_indices], targets[validation_indices], scale)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    training = {
        "trainer": "visual_trajectory_imitation",
        "vision": args.vision,
        "telemetry": args.telemetry,
        "samples": len(dataset.actions),
        "training_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "validation_split": "last fraction of each phase in each run",
        "max_alignment_error_ms": float(dataset.alignment_error_s.max() * 1000.0),
        "history": history,
        "validation": metrics,
        "elapsed_s": perf_counter() - start,
        "limitations": "Within-run visual behavior cloning; firmware stabilization only; not a general navigation policy.",
    }
    save_vision_action_policy(args.checkpoint, policy.eval(), training=training)
    report_path = args.checkpoint.with_suffix(".report.json")
    report_path.write_text(json.dumps(training, indent=2, sort_keys=True) + "\n")
    print(f"checkpoint={args.checkpoint}")
    print(f"report={report_path}")
    print(json.dumps(metrics, sort_keys=True))


@torch.no_grad()
def mse(policy, observations: torch.Tensor, targets: torch.Tensor) -> float:
    policy.eval()
    return float(F.mse_loss(policy(observations), targets))


@torch.no_grad()
def action_metrics(
    policy: CompactVisionActionPolicy,
    observations: torch.Tensor,
    targets: torch.Tensor,
    scale: VisionActionScale,
) -> dict[str, float]:
    policy.eval()
    predicted = scale.physical(policy(observations).numpy())
    expected = scale.physical(targets.numpy())
    error = np.abs(predicted - expected)
    return {
        "normalized_mse": float(np.mean((policy(observations).numpy() - targets.numpy()) ** 2)),
        "vx_mae_m_s": float(error[:, 0].mean()),
        "vy_mae_m_s": float(error[:, 1].mean()),
        "yawrate_mae_deg_s": float(error[:, 2].mean()),
        "vx_p95_m_s": float(np.percentile(error[:, 0], 95)),
        "vy_p95_m_s": float(np.percentile(error[:, 1], 95)),
        "yawrate_p95_deg_s": float(np.percentile(error[:, 2], 95)),
    }


if __name__ == "__main__":
    main()
