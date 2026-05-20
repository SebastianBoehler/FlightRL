from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from flightrl.sixdof import SixDofCrazyflieEnv, native_step_env
from flightrl.sixdof.policies import teacher_actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark 6-DoF Python/native env throughput across env counts")
    parser.add_argument("--env-counts", type=int, nargs="+", default=[1024, 4096, 8192, 16384])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", default="artifacts/replay/sixdof_native_benchmark_sweep.json")
    args = parser.parse_args()

    results = [benchmark_count(num_envs, args.steps) for num_envs in args.env_counts]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"steps": args.steps, "results": results, "best": max(results, key=lambda row: row["native_env_steps_per_second"])}
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"summary={output}")
    for row in results:
        print(
            f"num_envs={row['num_envs']} python_sps={row['python_steps_per_second']:.0f} "
            f"native_raw_sps={row['native_raw_steps_per_second']:.0f} native_env_sps={row['native_env_steps_per_second']:.0f}"
        )


def benchmark_count(num_envs: int, steps: int) -> dict[str, float]:
    py_env = SixDofCrazyflieEnv(num_envs=num_envs, seed=3, use_native_step=False)
    native_env = SixDofCrazyflieEnv(num_envs=num_envs, seed=3, use_native_step=True)
    actions = teacher_actions(py_env, task="position_yaw")
    python_sps = time_loop(lambda: py_env.step(actions), num_envs, steps)
    raw_sps = time_loop(lambda: native_step_env_once(native_env, actions), num_envs, steps)
    env_sps = time_loop(lambda: native_env.step(actions), num_envs, steps)
    return {
        "num_envs": num_envs,
        "python_steps_per_second": python_sps,
        "native_raw_steps_per_second": raw_sps,
        "native_env_steps_per_second": env_sps,
        "raw_speedup": raw_sps / python_sps,
        "env_speedup": env_sps / python_sps,
    }


def time_loop(fn, num_envs: int, steps: int) -> float:
    start = perf_counter()
    for _ in range(steps):
        fn()
    return (num_envs * steps) / (perf_counter() - start)


def native_step_env_once(env: SixDofCrazyflieEnv, actions: np.ndarray) -> None:
    native_step_env(env, actions)


def render_markdown(report: dict) -> str:
    lines = [
        "# 6-DoF Native Benchmark Sweep",
        "",
        "| envs | python sps | native raw sps | native env sps | env speedup |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['num_envs']} | {row['python_steps_per_second']:.0f} | "
            f"{row['native_raw_steps_per_second']:.0f} | {row['native_env_steps_per_second']:.0f} | {row['env_speedup']:.2f}x |"
        )
    best = report["best"]
    lines.extend(["", f"Best native env throughput: `{best['native_env_steps_per_second']:.0f}` steps/sec at `{best['num_envs']}` envs."])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
