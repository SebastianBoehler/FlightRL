from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from flightrl.sixdof import SixDofCrazyflieEnv, native_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Python vs native 6-DoF Crazyflie stepping")
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    actions = random_actions(args.num_envs, args.steps, args.seed)
    python_sps = benchmark_python(args.num_envs, actions, args.seed)
    native_sps = benchmark_native(args.num_envs, actions, args.seed)
    print(f"python_steps_per_second={python_sps:.0f}")
    print(f"native_steps_per_second={native_sps:.0f}")
    print(f"speedup={native_sps / max(python_sps, 1.0):.2f}x")


def random_actions(num_envs: int, steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.35, 0.35, size=(steps, num_envs, 4)).astype(np.float32)


def benchmark_python(num_envs: int, actions: np.ndarray, seed: int) -> float:
    env = SixDofCrazyflieEnv(num_envs=num_envs, seed=seed)
    env.reset(seed=seed)
    start = perf_counter()
    for action in actions:
        env.step(action)
    return actions.shape[0] * num_envs / (perf_counter() - start)


def benchmark_native(num_envs: int, actions: np.ndarray, seed: int) -> float:
    env = SixDofCrazyflieEnv(num_envs=num_envs, seed=seed)
    env.reset(seed=seed)
    start = perf_counter()
    for action in actions:
        native_step(env.position, env.velocity, env.quaternion, env.body_rates, env.ranges_m, action, env.dt)
    return actions.shape[0] * num_envs / (perf_counter() - start)


if __name__ == "__main__":
    main()
