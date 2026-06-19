from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flightrl.hardware.avoidance_policy import (
    RangerAvoidancePolicy,
    command_array,
    normalize_reading,
    reactive_clearance_command,
    sample_readings,
    teacher_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ranger-based obstacle avoidance setpoint policy")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/ranger_avoidance.pt")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--teacher", choices=("legacy", "reactive"), default="legacy")
    parser.add_argument("--clearance-m", type=float, default=0.45)
    parser.add_argument("--hard-clearance-m", type=float, default=0.10)
    parser.add_argument("--target-height-m", type=float, default=0.45)
    parser.add_argument("--max-speed-m-s", type=float, default=0.25)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    readings = sample_readings(args.samples, rng)
    observations = torch.from_numpy(np.stack([normalize_reading(reading) for reading in readings]))
    targets = torch.from_numpy(np.stack([command_array(build_teacher_command(reading, args)) for reading in readings]))
    model = RangerAvoidancePolicy(hidden_size=args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(observations.shape[0])
        losses = []
        for start in range(0, observations.shape[0], args.batch_size):
            indices = permutation[start : start + args.batch_size]
            prediction = model(observations[indices])
            loss = F.mse_loss(prediction, targets[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            print(f"epoch={epoch} loss={np.mean(losses):.6f}")

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden_size": args.hidden_size}, checkpoint)
    print(f"checkpoint={checkpoint}")


def build_teacher_command(reading, args):
    if args.teacher == "reactive":
        return reactive_clearance_command(
            reading,
            clearance_m=args.clearance_m,
            hard_clearance_m=args.hard_clearance_m,
            target_height_m=args.target_height_m,
            max_speed_m_s=args.max_speed_m_s,
        )
    return teacher_command(
        reading,
        min_distance_m=0.6,
        target_height_m=args.target_height_m,
        max_speed_m_s=args.max_speed_m_s,
    )


if __name__ == "__main__":
    main()
