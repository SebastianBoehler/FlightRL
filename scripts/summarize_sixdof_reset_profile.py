from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flightrl.sixdof import SixDofEnv, teacher_actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize simulated 6-DoF reset/profile ranger distribution.")
    parser.add_argument("--reset-profile", required=True)
    parser.add_argument("--task", default="obstacle_avoidance")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=0, help="Optional teacher rollout steps after reset.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--native-step", action="store_true")
    parser.add_argument("--live-envelope", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = summarize(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_report(report) + "\n")
    print(f"profile_report={output}")
    print(f"hmin_median={report['reset_ranges']['hmin']['median']:.3f} hmin_p10={report['reset_ranges']['hmin']['p10']:.3f}")


def summarize(args: argparse.Namespace) -> dict:
    env = SixDofEnv(
        num_envs=args.num_envs,
        seed=args.seed,
        task=args.task,
        reset_profile=args.reset_profile,
        use_native_step=args.native_step,
    )
    env.reset(seed=args.seed)
    reset_ranges = range_summary(env.ranges_m)
    rollout_ranges = None
    if args.steps > 0:
        samples = []
        for _ in range(args.steps):
            action = teacher_actions(env, task=args.task)
            env.step(action)
            samples.append(env.ranges_m.copy())
        rollout_ranges = range_summary(np.concatenate(samples, axis=0))
    live = json.loads(Path(args.live_envelope).read_text()) if args.live_envelope else None
    return {
        "reset_profile": args.reset_profile,
        "task": args.task,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "native_step": args.native_step,
        "reset_ranges": reset_ranges,
        "rollout_ranges": rollout_ranges,
        "live_comparison": compare_live(reset_ranges, live) if live else None,
    }


def range_summary(ranges_m: np.ndarray) -> dict:
    horizontal = ranges_m[:, :4]
    values = {
        "front": horizontal[:, 0],
        "back": horizontal[:, 1],
        "left": horizontal[:, 2],
        "right": horizontal[:, 3],
        "hmin": np.min(horizontal, axis=1),
        "zrange": ranges_m[:, 5],
        "up": ranges_m[:, 4],
    }
    return {key: quantiles(array) for key, array in values.items()}


def quantiles(values: np.ndarray) -> dict:
    return {
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def compare_live(reset_ranges: dict, live: dict) -> dict:
    live_hmin = live.get("hmin", {})
    sim_hmin = reset_ranges["hmin"]
    return {
        "hmin_p01_delta_m": float(sim_hmin["p01"] - live_hmin.get("p01", sim_hmin["p01"])),
        "hmin_p10_delta_m": float(sim_hmin["p10"] - live_hmin.get("p10", sim_hmin["p10"])),
        "hmin_median_delta_m": float(sim_hmin["median"] - live_hmin.get("median", sim_hmin["median"])),
    }


def render_report(report: dict) -> str:
    lines = [
        "# 6-DoF Reset Profile Summary",
        "",
        f"- Reset profile: `{report['reset_profile']}`",
        f"- Task: `{report['task']}`",
        f"- Samples: `{report['num_envs']}`",
        f"- Teacher rollout steps: `{report['steps']}`",
        "",
        "## Reset Horizontal Minimum",
        "",
        render_quantiles(report["reset_ranges"]["hmin"]),
    ]
    if report["rollout_ranges"] is not None:
        lines.extend(["", "## Rollout Horizontal Minimum", "", render_quantiles(report["rollout_ranges"]["hmin"])])
    if report["live_comparison"] is not None:
        delta = report["live_comparison"]
        lines.extend(
            [
                "",
                "## Live Delta",
                "",
                f"- hmin p01 delta m: `{delta['hmin_p01_delta_m']:.3f}`",
                f"- hmin p10 delta m: `{delta['hmin_p10_delta_m']:.3f}`",
                f"- hmin median delta m: `{delta['hmin_median_delta_m']:.3f}`",
            ]
        )
    return "\n".join(lines)


def render_quantiles(item: dict) -> str:
    return (
        f"- min `{item['min']:.3f}`, p01 `{item['p01']:.3f}`, p10 `{item['p10']:.3f}`, "
        f"median `{item['median']:.3f}`, p90 `{item['p90']:.3f}`, p99 `{item['p99']:.3f}`, max `{item['max']:.3f}`"
    )


if __name__ == "__main__":
    main()
