from __future__ import annotations

import argparse

from flightrl.plotting import plot_trajectory
from flightrl.rollout import load_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a saved FlightRL trajectory")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="artifacts/trajectories/trajectory.png")
    args = parser.parse_args()

    output = plot_trajectory(load_rollout(args.input), args.output)
    print(output)


if __name__ == "__main__":
    main()
