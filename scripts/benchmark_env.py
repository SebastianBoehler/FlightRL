from __future__ import annotations

import argparse
import time

import numpy as np

from flightrl import load_config, make_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark native environment steps/sec")
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=512)
    args = parser.parse_args()

    env = make_env(load_config(args.config))
    env.reset(seed=0)
    cache = np.random.uniform(-1.0, 1.0, size=(64, env.num_agents, env.single_action_space.shape[0])).astype(np.float32)
    start = time.perf_counter()
    for idx in range(args.steps):
        env.step(cache[idx % len(cache)])
    elapsed = time.perf_counter() - start
    sps = int((env.num_agents * args.steps) / elapsed)
    env.close()
    print({"steps_per_second": sps, "elapsed_seconds": round(elapsed, 4)})


if __name__ == "__main__":
    main()
