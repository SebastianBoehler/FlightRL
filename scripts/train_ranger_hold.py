from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flightrl.hardware.hold_policy import (
    HOLD_OBSERVATION_DIM,
    HOLD_OUTPUT_SCALE,
    RangerHoldPolicy,
    normalize_hold_state,
    normalized_hold_command_array,
    sample_hold_states,
    teacher_hold_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Crazyflie ranger/attitude hold setpoint policy")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/ranger_hold.pt")
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    states = sample_hold_states(args.samples, rng)
    observations = torch.from_numpy(np.stack([normalize_hold_state(state) for state in states]))
    targets = torch.from_numpy(np.stack([normalized_hold_command_array(teacher_hold_command(state)) for state in states]))
    model = RangerHoldPolicy(hidden_size=args.hidden_size)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            print(f"epoch={epoch} loss={np.mean(losses):.6f}")

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_size": args.hidden_size,
            "observation_dim": HOLD_OBSERVATION_DIM,
            "output_scale": HOLD_OUTPUT_SCALE.tolist(),
            "teacher": "reactive_ranger_position_attitude_hold",
        },
        checkpoint,
    )
    print(f"checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
