from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flightrl.sixdof import BoxRoom, SixDofCrazyflieEnv, teacher_actions
from flightrl.sixdof.env import quat_to_yaw


RANGE_SIGNALS = ("range.front", "range.back", "range.left", "range.right", "range.up", "range.zrange")
STATE_SIGNALS = (
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stateEstimate.vx",
    "stateEstimate.vy",
    "stateEstimate.vz",
    "stabilizer.yaw",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Python-vs-native 6-DoF dynamics parity report")
    parser.add_argument("--task", default="position_yaw")
    parser.add_argument("--reset-profile", action="append", default=None)
    parser.add_argument("--room-report", default=None)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=333)
    parser.add_argument("--action-source", choices=("random", "teacher", "zero"), default="random")
    parser.add_argument("--action-scale", type=float, default=0.25)
    parser.add_argument("--output", default="artifacts/replay/sixdof_native_parity.json")
    args = parser.parse_args()

    room = load_room_report(args.room_report) if args.room_report else None
    reset_profiles = args.reset_profile or ["position_yaw_medium"]
    profiles = [run_profile(args, profile, room) for profile in reset_profiles]
    report = {
        "task": args.task,
        "reset_profiles": reset_profiles,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "seed": args.seed,
        "action_source": args.action_source,
        "action_scale": args.action_scale,
        "room_report": args.room_report,
        "profiles": profiles,
        "aligned": aggregate_aligned(profiles),
        "safety": "Native parity validates simulator implementations only; it is not a live-flight approval.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"native_parity={output}")
    print(f"markdown={output.with_suffix('.md')}")


def run_profile(args: argparse.Namespace, profile: str, room: BoxRoom | None) -> dict:
    py_env = SixDofCrazyflieEnv(args.num_envs, seed=args.seed, task=args.task, room=room, use_native_step=False, reset_profile=profile)
    native_env = SixDofCrazyflieEnv(args.num_envs, seed=args.seed, task=args.task, room=room, use_native_step=True, reset_profile=profile)
    rng = np.random.default_rng(args.seed + 10_000)
    errors = {key: [] for key in (*STATE_SIGNALS, *RANGE_SIGNALS)}
    terminal_mismatches = 0
    truncation_mismatches = 0
    for _ in range(args.steps):
        actions = actions_for(py_env, rng, args.action_source, args.action_scale)
        py_env.step(actions)
        native_env.step(actions)
        collect_errors(errors, py_env, native_env)
        terminal_mismatches += int(np.count_nonzero(py_env.terminals != native_env.terminals))
        truncation_mismatches += int(np.count_nonzero(py_env.truncations != native_env.truncations))
        done = (py_env.terminals | py_env.truncations | native_env.terminals | native_env.truncations).astype(bool)
        if np.any(done):
            py_env.reset_done(done)
            native_env.reset_done(done)
    signals = {key: error_metrics(np.concatenate(values)) for key, values in errors.items() if values}
    return {
        "reset_profile": profile,
        "samples": args.num_envs * args.steps,
        "duration_s": args.steps * py_env.dt,
        "terminal_mismatches": terminal_mismatches,
        "truncation_mismatches": truncation_mismatches,
        "signals": signals,
    }


def actions_for(env: SixDofCrazyflieEnv, rng: np.random.Generator, source: str, scale: float) -> np.ndarray:
    if source == "zero":
        return np.zeros((env.num_envs, 4), dtype=np.float32)
    if source == "teacher":
        return teacher_actions(env, task=env.task).astype(np.float32)
    return np.clip(rng.normal(0.0, scale, size=(env.num_envs, 4)), -1.0, 1.0).astype(np.float32)


def collect_errors(errors: dict[str, list[np.ndarray]], py_env: SixDofCrazyflieEnv, native_env: SixDofCrazyflieEnv) -> None:
    py_state = signal_arrays(py_env)
    native_state = signal_arrays(native_env)
    for key in errors:
        errors[key].append(native_state[key] - py_state[key])


def signal_arrays(env: SixDofCrazyflieEnv) -> dict[str, np.ndarray]:
    ranges_mm = env.ranges_m * 1000.0
    return {
        "stateEstimate.x": env.position[:, 0],
        "stateEstimate.y": env.position[:, 1],
        "stateEstimate.z": env.position[:, 2],
        "stateEstimate.vx": env.velocity[:, 0],
        "stateEstimate.vy": env.velocity[:, 1],
        "stateEstimate.vz": env.velocity[:, 2],
        "stabilizer.yaw": quat_to_yaw(env.quaternion) * 180.0 / np.pi,
        "range.front": ranges_mm[:, 0],
        "range.back": ranges_mm[:, 1],
        "range.left": ranges_mm[:, 2],
        "range.right": ranges_mm[:, 3],
        "range.up": ranges_mm[:, 4],
        "range.zrange": ranges_mm[:, 5],
    }


def error_metrics(error: np.ndarray) -> dict[str, float]:
    return {"samples": int(error.size), "rmse": rmse(error), "mae": float(np.mean(np.abs(error))), "max_abs": float(np.max(np.abs(error)))}


def rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error.astype(np.float64)))))


def aggregate_aligned(profiles: list[dict]) -> dict:
    signals = {}
    for key in (*STATE_SIGNALS, *RANGE_SIGNALS):
        worst = max(profiles, key=lambda item: item["signals"][key]["rmse"])
        signals[key] = dict(worst["signals"][key])
        signals[key]["worst_profile"] = worst["reset_profile"]
    return {"samples": int(sum(item["samples"] for item in profiles)), "overlap_duration_s": float(sum(item["duration_s"] for item in profiles)), "signals": signals}


def load_room_report(path: str | None) -> BoxRoom | None:
    report = json.loads(Path(path).read_text())
    estimate = report.get("room_estimate")
    if not estimate:
        raise SystemExit(f"room report has no room_estimate: {path}")
    return BoxRoom(
        x_min=float(estimate["x_min"]),
        x_max=float(estimate["x_max"]),
        y_min=float(estimate["y_min"]),
        y_max=float(estimate["y_max"]),
        z_min=float(estimate["z_min"]),
        z_max=float(estimate["z_max"]),
        max_range_m=float(estimate.get("max_range_m", 4.0)),
    )


def render_markdown(report: dict) -> str:
    lines = [
        "# 6-DoF Native Parity",
        "",
        "| profile | samples | terminal mismatches | worst state RMSE | worst range RMSE mm |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for profile in report["profiles"]:
        state = max(metric["rmse"] for key, metric in profile["signals"].items() if key.startswith("stateEstimate."))
        ranges = max(metric["rmse"] for key, metric in profile["signals"].items() if key.startswith("range."))
        lines.append(f"| {profile['reset_profile']} | {profile['samples']} | {profile['terminal_mismatches']} | {state:.8g} | {ranges:.8g} |")
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
