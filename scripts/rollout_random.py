from __future__ import annotations

import argparse

from flightrl import load_config, make_env
from flightrl.plotting import plot_trajectory
from flightrl.rollout import collect_rollout, save_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a random-policy trajectory rollout")
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--output", default="artifacts/trajectories/random_rollout.csv")
    parser.add_argument("--render-mode", choices=["human", "rgb_array"], default=None)
    args = parser.parse_args()

    env = make_env(load_config(args.config), render_mode=args.render_mode)
    trace = collect_rollout(env, args.steps)
    output_path = save_rollout(trace, args.output)
    plot_path = plot_trajectory(trace, output_path.with_suffix(".png"))
    env.close()
    print(output_path)
    print(plot_path)


if __name__ == "__main__":
    main()
