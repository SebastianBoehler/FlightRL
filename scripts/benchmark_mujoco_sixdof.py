from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from flightrl.mujoco import MuJoCoCrazyflieEnv, is_mujoco_available
from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.policies import teacher_actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FlightRL native 6-DoF stepping against the optional MuJoCo backend")
    parser.add_argument("--env-counts", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sensor-profile", default=None)
    parser.add_argument("--output", default="artifacts/replay/mujoco_sixdof_benchmark.json")
    parser.add_argument("--allow-missing-mujoco", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not is_mujoco_available():
        if not args.allow_missing_mujoco:
            raise SystemExit("MuJoCo is not installed. Run: python -m pip install -e '.[mujoco]' --no-build-isolation")
        report = {"status": "missing_mujoco", "install": "python -m pip install -e '.[mujoco]' --no-build-isolation"}
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        output.with_suffix(".md").write_text("# MuJoCo 6-DoF Benchmark\n\nMuJoCo is not installed.\n")
        print(f"summary={output}")
        return

    results = [benchmark_count(num_envs, args.steps, args.seed, args.sensor_profile) for num_envs in args.env_counts]
    report = {"status": "ok", "steps": args.steps, "sensor_profile": args.sensor_profile, "results": results, "best_native": max(results, key=lambda row: row["native_env_sps"])}
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"summary={output}")
    for row in results:
        print(
            f"num_envs={row['num_envs']} python_sps={row['python_sps']:.0f} "
            f"native_env_sps={row['native_env_sps']:.0f} mujoco_sps={row['mujoco_sps']:.0f} "
            f"native_vs_mujoco={row['native_vs_mujoco']:.2f}x"
        )


def benchmark_count(num_envs: int, steps: int, seed: int, sensor_profile: str | None) -> dict[str, float]:
    py_env = SixDofCrazyflieEnv(num_envs=num_envs, seed=seed, use_native_step=False, sensor_profile=sensor_profile)
    native_env = SixDofCrazyflieEnv(num_envs=num_envs, seed=seed, use_native_step=True, sensor_profile=sensor_profile)
    mujoco_env = MuJoCoCrazyflieEnv(num_envs=num_envs, seed=seed, sensor_profile=sensor_profile)
    actions = teacher_actions(py_env, task="position_yaw")
    python_sps = time_loop(lambda: py_env.step(actions), num_envs, steps)
    native_sps = time_loop(lambda: native_env.step(actions), num_envs, steps)
    mujoco_sps = time_loop(lambda: mujoco_env.step(actions), num_envs, steps)
    return {
        "num_envs": num_envs,
        "python_sps": python_sps,
        "native_env_sps": native_sps,
        "mujoco_sps": mujoco_sps,
        "native_vs_python": native_sps / max(python_sps, 1.0),
        "native_vs_mujoco": native_sps / max(mujoco_sps, 1.0),
        "mujoco_vs_python": mujoco_sps / max(python_sps, 1.0),
    }


def time_loop(fn, num_envs: int, steps: int) -> float:
    start = perf_counter()
    for _ in range(steps):
        fn()
    return (num_envs * steps) / (perf_counter() - start)


def render_markdown(report: dict) -> str:
    lines = [
        "# MuJoCo 6-DoF Benchmark",
        "",
        "Same action, observation, reward, and task shape across Python six-DoF, native C env stepping, and MuJoCo-backed rigid-body stepping.",
        "",
        "| envs | python sps | native env sps | MuJoCo sps | native / MuJoCo | MuJoCo / python |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['num_envs']} | {row['python_sps']:.0f} | {row['native_env_sps']:.0f} | "
            f"{row['mujoco_sps']:.0f} | {row['native_vs_mujoco']:.2f}x | {row['mujoco_vs_python']:.2f}x |"
        )
    best = report["best_native"]
    lines.append("")
    lines.append(f"Best native throughput: `{best['native_env_sps']:.0f}` steps/sec at `{best['num_envs']}` envs.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
