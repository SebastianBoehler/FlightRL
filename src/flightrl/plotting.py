from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_trajectory(trace: list[dict[str, float]], output_path: str | Path) -> Path:
    xs = [row["x"] for row in trace]
    zs = [row["z"] for row in trace]
    tx = [row["target_x"] for row in trace]
    tz = [row["target_z"] for row in trace]
    rewards = [row["reward"] for row in trace]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(xs, zs, label="drone")
    axes[0].scatter(tx, tz, s=10, c="tab:orange", label="target")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")
    axes[0].set_title("Trajectory")
    axes[0].legend()

    axes[1].plot(rewards, label="reward")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("reward")
    axes[1].set_title("Reward Trace")
    axes[1].legend()

    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output
