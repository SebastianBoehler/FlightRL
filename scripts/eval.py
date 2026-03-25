from __future__ import annotations

import argparse

from flightrl import load_config
from flightrl.plotting import plot_trajectory
from flightrl.rollout import collect_rollout, save_rollout
from flightrl.training import create_env_and_policy, load_policy_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a FlightRL policy")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--output", default="artifacts/trajectories/eval.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    env, policy = create_env_and_policy(config)
    if args.checkpoint:
        load_policy_checkpoint(policy, args.checkpoint, device=config.training.device)

    trace = collect_rollout(env, steps=args.steps, policy=policy)
    output_path = save_rollout(trace, args.output)
    plot_path = plot_trajectory(trace, output_path.with_suffix(".png"))
    env.close()
    print(output_path)
    print(plot_path)


if __name__ == "__main__":
    main()
