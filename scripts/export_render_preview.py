from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from flightrl import load_config, make_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a clean renderer preview frame")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-scale", type=float, default=0.3)
    args = parser.parse_args()

    config = load_config(args.config)
    env = make_env(config, seed=args.seed, render_mode="rgb_array")
    rng = np.random.default_rng(args.seed)
    env.reset(seed=args.seed)
    frame = env.render()
    for _ in range(args.steps):
        action = rng.uniform(
            -args.action_scale,
            args.action_scale,
            size=(env.num_agents, config.action_dim),
        ).astype(np.float32)
        env.step(action)
        frame = env.render()
    env.close()

    plt.imsave(args.output, frame)
    print(args.output)


if __name__ == "__main__":
    main()
