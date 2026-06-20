from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_sixdof_export import SIXDOF_NATIVE_FILES, export_sixdof_puffer4_assets


REQUIRED_BINDING_TOKENS = (
    "#define OBS_SIZE 28",
    "#define NUM_ATNS 4",
    '#include "vecenv.h"',
    "flightrl_sixdof_step_env_context_batch",
    'dict_get(kwargs, "room_x_min")',
    'dict_get(kwargs, "mass_kg")',
    'dict_get(kwargs, "task_id")',
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and validate the Crazyflie 6-DoF PufferLib/Ocean scaffold")
    parser.add_argument("--pufferlib-root", default="artifacts/puffer4_sixdof_export")
    parser.add_argument("--env-name", default="flightrl_sixdof_report")
    parser.add_argument("--output", default="artifacts/replay/sixdof_puffer_export_report.json")
    parser.add_argument("--total-agents", type=int, default=4096)
    parser.add_argument("--num-buffers", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-profile", default=None)
    parser.add_argument("--task", default="position_yaw")
    parser.add_argument("--reward-mode", default="env")
    parser.add_argument("--reset-profile", default="broad")
    args = parser.parse_args()

    report = build_report(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"puffer_export={output}")
    print(f"markdown={output.with_suffix('.md')}")
    if not report["passed"]:
        raise SystemExit(1)


def build_report(args: argparse.Namespace) -> dict:
    settings = Puffer4ExportSettings(
        env_name=args.env_name,
        total_agents=args.total_agents,
        num_buffers=args.num_buffers,
        num_threads=args.num_threads,
        policy_hidden_size=args.hidden_size,
        train_seed=args.seed,
        sim_profile=args.sim_profile,
        task=args.task,
        reward_mode=args.reward_mode,
        reset_profile=args.reset_profile,
    )
    result = export_sixdof_puffer4_assets(args.pufferlib_root, settings=settings)
    files = inspect_files(result.env_dir, result.config_path)
    config = parse_ini(result.config_path)
    checks = validate(files, config, args)
    return {
        "env_name": result.env_name,
        "pufferlib_root": str(Path(args.pufferlib_root).expanduser().resolve()),
        "env_dir": str(result.env_dir),
        "config_path": str(result.config_path),
        "files": files,
        "config": config,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "safety": "PufferLib export scaffold validation only; this is not a trained policy or hardware deployment approval.",
    }


def inspect_files(env_dir: Path, config_path: Path) -> dict:
    paths = {"binding.c": env_dir / "binding.c", "config": config_path}
    paths.update({filename: env_dir / filename for filename in SIXDOF_NATIVE_FILES})
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "lines": line_count(path) if path.exists() else 0,
            "required_tokens": token_hits(path, REQUIRED_BINDING_TOKENS) if name == "binding.c" and path.exists() else {},
        }
        for name, path in paths.items()
    }


def validate(files: dict, config: dict, args: argparse.Namespace) -> list[dict]:
    checks = [
        check("files_present", all(file["exists"] and file["bytes"] > 0 for file in files.values())),
        check("binding_tokens", all(files["binding.c"]["required_tokens"].values())),
        check("config_env_name", config.get("base", {}).get("env_name") == args.env_name),
        check("config_total_agents", int_value(config, "vec", "total_agents") == args.total_agents),
        check("config_num_buffers", int_value(config, "vec", "num_buffers") == args.num_buffers),
        check("config_num_threads", int_value(config, "vec", "num_threads") == args.num_threads),
        check("config_hidden_size", int_value(config, "policy", "hidden_size") == args.hidden_size),
        check("room_bounds_present", all(key in config.get("env", {}) for key in ("room_x_min", "room_x_max", "room_y_min", "room_y_max", "room_z_min", "room_z_max", "max_range_m"))),
        check("native_step_include_copied", files.get("native_sixdof_step.inc", {}).get("exists", False)),
        check("native_step_include_referenced", file_contains(files, "native_sixdof.c", '#include "native_sixdof_step.inc"')),
        check(
            "physics_knobs_present",
            all(
                key in config.get("env", {})
                for key in (
                    "mass_kg",
                    "gravity_m_s2",
                    "linear_drag",
                    "rate_tau_s",
                    "thrust_scale",
                    "max_rate_roll",
                    "max_rate_pitch",
                    "max_rate_yaw",
                    "motor_tau_s",
                )
            ),
        ),
        check(
            "sensor_profile_knobs_present",
            all(key in config.get("env", {}) for key in ("range_noise_std_m", "range_dropout_prob", "action_lag_s")),
        ),
        check("task_id_present", "task_id" in config.get("env", {})),
        check("reward_mode_present", "reward_mode" in config.get("env", {})),
        check(
            "reset_profile_knobs_present",
            all(
                key in config.get("env", {})
                for key in (
                    "near_wall_probability",
                    "reset_z_min",
                    "reset_z_max",
                    "target_z_min",
                    "target_z_max",
                    "target_xy_offset_abs",
                    "target_yaw_offset_abs",
                )
            ),
        ),
    ]
    return checks


def check(name: str, passed: bool) -> dict[str, bool | str]:
    return {"name": name, "passed": bool(passed)}


def int_value(config: dict, section: str, key: str) -> int | None:
    value = config.get(section, {}).get(key)
    return int(value) if value is not None else None


def file_contains(files: dict, name: str, token: str) -> bool:
    path = Path(files.get(name, {}).get("path", ""))
    return path.exists() and token in path.read_text()


def parse_ini(path: Path) -> dict[str, dict[str, str]]:
    current = None
    parsed: dict[str, dict[str, str]] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            parsed[current] = {}
        elif current and "=" in line:
            key, value = line.split("=", 1)
            parsed[current][key.strip()] = value.strip()
    return parsed


def line_count(path: Path) -> int:
    return len(path.read_text().splitlines())


def token_hits(path: Path, tokens: tuple[str, ...]) -> dict[str, bool]:
    text = path.read_text()
    return {token: token in text for token in tokens}


def render_markdown(report: dict) -> str:
    lines = ["# 6-DoF PufferLib Export Report", "", f"Passed: `{report['passed']}`", "", "| check | passed |", "| --- | ---: |"]
    lines.extend(f"| {check['name']} | {check['passed']} |" for check in report["checks"])
    lines.extend(["", "## Files", "", "| file | exists | bytes | lines |", "| --- | ---: | ---: | ---: |"])
    lines.extend(f"| {name} | {meta['exists']} | {meta['bytes']} | {meta['lines']} |" for name, meta in report["files"].items())
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
