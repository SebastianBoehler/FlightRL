from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flightrl.sixdof import SixDofPolicy
from flightrl.sixdof.observation import OBSERVATION_MODES, observation_dim
from flightrl.sixdof.tasks import parse_task_spec, task_observation_dim


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a teacher-residual 6-DoF checkpoint scaffold")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="circle")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--observation-mode", default="base", choices=OBSERVATION_MODES)
    parser.add_argument("--residual-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--zero-weights", action="store_true", help="Initialize the residual network to output exactly zero.")
    args = parser.parse_args()

    tasks = parse_task_spec(args.task)
    input_dim = observation_dim(28 + task_observation_dim(tasks), args.observation_mode)
    torch.manual_seed(args.seed)
    model = SixDofPolicy(hidden_size=args.hidden_size, input_dim=input_dim)
    if args.zero_weights:
        for parameter in model.parameters():
            parameter.data.zero_()
    output = Path(args.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "task": ",".join(tasks),
            "tasks": list(tasks),
            "task_conditioned": len(tasks) > 1,
            "hidden_size": args.hidden_size,
            "observation_dim": input_dim,
            "base_observation_dim": 28,
            "observation_mode": args.observation_mode,
            "action_dim": 4,
            "controller": "teacher_residual",
            "residual_scale": args.residual_scale,
            "trainer": "residual_scaffold",
            "note": "Teacher-residual checkpoint scaffold; simulation gate only and not approved for live hardware.",
        },
        output,
    )
    print(f"checkpoint={output}")


if __name__ == "__main__":
    main()
