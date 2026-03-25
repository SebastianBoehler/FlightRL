from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from flightrl.rollout import load_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reward traces from two rollout files")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", default="artifacts/trajectories/reward_compare.png")
    args = parser.parse_args()

    left = load_rollout(args.left)
    right = load_rollout(args.right)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([row["reward"] for row in left], label=Path(args.left).stem)
    ax.plot([row["reward"] for row in right], label=Path(args.right).stem)
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.set_title("Reward Comparison")
    ax.legend()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
