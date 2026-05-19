from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flightrl.hardware.avoidance_policy import RangerAvoidancePolicy, command_array, normalize_reading, sample_readings, teacher_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ranger-based obstacle avoidance setpoint policy")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/ranger_avoidance.pt")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    readings = sample_readings(args.samples, rng)
    observations = torch.from_numpy(np.stack([normalize_reading(reading) for reading in readings]))
    targets = torch.from_numpy(np.stack([command_array(teacher_command(reading)) for reading in readings]))
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


if __name__ == "__main__":
    main()
