from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def build_live_system_id_report(*, flight_logs: list[Path], base_profile: Path | None = None, name: str) -> dict[str, Any]:
    run_reports = [analyze_run(path) for path in flight_logs]
    response = aggregate_response(run_reports)
    sensor_profile = calibrated_sensor_profile(base_profile, response, name)
    return {
        "name": name,
        "inputs": {"flight_logs": [str(path) for path in flight_logs], "base_profile": str(base_profile) if base_profile else None},
        "summary": {
            "runs": len(run_reports),
            "rows": int(sum(run["rows"] for run in run_reports)),
            "profile_ready": bool(run_reports),
            "battery_v_min": min((run["battery"]["min_v"] for run in run_reports if run["battery"]["min_v"] is not None), default=None),
            "battery_level_min": min((run["battery"]["min_level"] for run in run_reports if run["battery"]["min_level"] is not None), default=None),
        },
        "response": response,
        "runs": run_reports,
        "sensor_profile": sensor_profile,
        "safety": "Offline system-identification profile only; live deployment still requires replay, shadow, and hardware gates.",
    }


def analyze_run(path: Path) -> dict[str, Any]:
    rows = read_rows(path)
    time_s = np.asarray([value(row, "host_time_s") for row in rows], dtype=np.float64)
    command = np.asarray([[value(row, "vx_m_s"), value(row, "vy_m_s")] for row in rows], dtype=np.float64)
    velocity = np.asarray([body_velocity(row) for row in rows], dtype=np.float64)
    dt_s = median_dt(time_s)
    lag = fit_lag(command, velocity, dt_s)
    tau = fit_tau(command, velocity, dt_s, lag["lag_steps"], lag["gain"])
    return {
        "log": str(path),
        "rows": len(rows),
        "duration_s": float(time_s[-1] - time_s[0]) if len(time_s) > 1 else 0.0,
        "sample_period_s": dt_s,
        "command": vector_summary(command),
        "body_velocity": vector_summary(velocity),
        "tracking": {**lag, **tau},
        "acceleration": acceleration_summary(velocity, time_s),
        "ranges": range_summary(rows),
        "battery": battery_summary(rows),
    }


def fit_lag(command: np.ndarray, velocity: np.ndarray, dt_s: float) -> dict[str, float | int | None]:
    mask = np.linalg.norm(command, axis=1) > 0.10
    best: dict[str, float | int | None] = {"lag_steps": 0, "lag_s": 0.0, "gain": 0.0, "rmse_m_s": None, "samples": int(np.sum(mask))}
    if len(command) < 5 or int(np.sum(mask)) < 20:
        return best
    max_steps = min(int(round(0.35 / max(dt_s, 1e-4))), len(command) // 3)
    for lag_steps in range(max_steps + 1):
        cmd = command[: len(command) - lag_steps or None]
        vel = velocity[lag_steps:]
        valid = mask[: len(cmd)]
        if int(np.sum(valid)) < 20:
            continue
        cmd_flat = cmd[valid].reshape(-1)
        vel_flat = vel[valid].reshape(-1)
        denom = float(np.dot(cmd_flat, cmd_flat))
        gain = float(np.dot(cmd_flat, vel_flat) / denom) if denom > 1e-9 else 0.0
        error = gain * cmd_flat - vel_flat
        rmse = float(np.sqrt(np.mean(error * error)))
        if best["rmse_m_s"] is None or rmse < float(best["rmse_m_s"]):
            best = {"lag_steps": lag_steps, "lag_s": float(lag_steps * dt_s), "gain": gain, "rmse_m_s": rmse, "samples": int(np.sum(valid))}
    return best


def fit_tau(command: np.ndarray, velocity: np.ndarray, dt_s: float, lag_steps: int | None, gain: float | None) -> dict[str, float | None]:
    if len(command) < 5 or gain is None:
        return {"alpha": None, "tau_s": None}
    lag = int(lag_steps or 0)
    cmd = command[: len(command) - lag - 1 or None]
    vel = velocity[lag : len(command) - 1]
    next_vel = velocity[lag + 1 :]
    target_delta = gain * cmd - vel
    observed_delta = next_vel - vel
    mask = np.linalg.norm(cmd, axis=1) > 0.10
    if int(np.sum(mask)) < 20:
        return {"alpha": None, "tau_s": None}
    x = target_delta[mask].reshape(-1)
    y = observed_delta[mask].reshape(-1)
    denom = float(np.dot(x, x))
    alpha = float(np.clip(np.dot(x, y) / denom, 1e-4, 1.0)) if denom > 1e-9 else None
    tau = float(dt_s * (1.0 - alpha) / alpha) if alpha is not None else None
    return {"alpha": alpha, "tau_s": tau}


def calibrated_sensor_profile(base_profile: Path | None, response: dict[str, Any], name: str) -> dict[str, Any]:
    profile = load_sensor_profile(base_profile)
    # The response fit is plant dynamics evidence; encoding it as command smoothing would double-count dynamics.
    profile["name"] = name
    profile["enabled"] = any(float(profile.get(key, 0.0) or 0.0) > 0.0 for key in profile if key not in {"name", "enabled"})
    return profile


def aggregate_response(runs: list[dict[str, Any]]) -> dict[str, Any]:
    lag_s = [run["tracking"]["lag_s"] for run in runs if run["tracking"]["lag_s"] is not None]
    tau_s = [run["tracking"]["tau_s"] for run in runs if run["tracking"]["tau_s"] is not None]
    gain = [run["tracking"]["gain"] for run in runs if run["tracking"]["gain"] is not None]
    rmse = [run["tracking"]["rmse_m_s"] for run in runs if run["tracking"]["rmse_m_s"] is not None]
    return {
        "lag_s": quantiles(lag_s),
        "tau_s": quantiles(tau_s),
        "gain": quantiles(gain),
        "rmse_m_s": quantiles(rmse),
        "runs": [{"log": run["log"], "tracking": run["tracking"]} for run in runs],
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def body_velocity(row: dict[str, str]) -> tuple[float, float]:
    vx = value(row, "stateEstimate.vx")
    vy = value(row, "stateEstimate.vy")
    yaw = math.radians(value(row, "stabilizer.yaw", "stateEstimate.yaw"))
    cy, sy = math.cos(yaw), math.sin(yaw)
    return cy * vx + sy * vy, -sy * vx + cy * vy


def acceleration_summary(velocity: np.ndarray, time_s: np.ndarray) -> dict[str, float | None]:
    if len(velocity) < 3:
        return quantiles([])
    dt = np.diff(time_s)
    dv = np.diff(velocity, axis=0)
    valid = (dt > 1e-4) & (dt < 0.2)
    accel = np.linalg.norm(dv[valid] / dt[valid, None], axis=1)
    return quantiles(accel)


def range_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    axes = ("front", "back", "left", "right", "up", "zrange")
    return {axis: range_axis_summary(rows, f"range.{axis}") for axis in axes}


def range_axis_summary(rows: list[dict[str, str]], column: str) -> dict[str, float | int]:
    values = [raw for row in rows if (raw := maybe_value(row, column)) is not None]
    present = [raw / 1000.0 for raw in values if raw < 32000.0]
    return {"dropout_fraction": 1.0 - len(present) / len(values) if values else 0.0, **quantiles(present)}


def battery_summary(rows: list[dict[str, str]]) -> dict[str, float | None]:
    vbat = [raw for row in rows if (raw := maybe_value(row, "pm.vbat")) is not None]
    level = [raw for row in rows if (raw := maybe_value(row, "pm.batteryLevel")) is not None]
    return {"min_v": min(vbat) if vbat else None, "min_level": min(level) if level else None, "vbat": quantiles(vbat)}


def vector_summary(values: np.ndarray) -> dict[str, float | None]:
    speed = np.linalg.norm(values, axis=1) if len(values) else np.asarray([])
    return quantiles(speed)


def median_dt(time_s: np.ndarray) -> float:
    if len(time_s) < 2:
        return 0.01
    diffs = np.diff(np.sort(time_s))
    diffs = diffs[(diffs > 0.0) & (diffs < 0.2)]
    return float(np.median(diffs)) if len(diffs) else 0.01


def quantiles(series: list[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(series, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"min": None, "p05": None, "median": None, "p95": None, "max": None}
    return {
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def load_sensor_profile(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "state_noise_std_m": 0.0,
            "velocity_noise_std_m_s": 0.0,
            "body_rate_noise_std_rad_s": 0.0,
            "range_noise_std_m": 0.0,
            "range_dropout_prob": 0.0,
            "action_lag_s": 0.0,
        }
    data = json.loads(path.read_text())
    return dict(data.get("sensor_profile", data))


def value(row: dict[str, str], key: str, fallback: str | None = None) -> float:
    raw = row.get(key)
    if (raw is None or raw == "") and fallback is not None:
        raw = row.get(fallback)
    try:
        return float(raw if raw not in {None, ""} else 0.0)
    except (TypeError, ValueError):
        return 0.0


def maybe_value(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key)
    try:
        return float(raw) if raw not in {None, ""} else None
    except (TypeError, ValueError):
        return None


def render_markdown(report: dict[str, Any]) -> str:
    response = report["response"]
    profile = report["sensor_profile"]
    lines = [
        "# Live System Identification",
        "",
        f"- Runs: `{report['summary']['runs']}`",
        f"- Rows: `{report['summary']['rows']}`",
        f"- Min battery: `{report['summary']['battery_v_min']}` V / `{report['summary']['battery_level_min']}` %",
        f"- Response lag median: `{_fmt(response['lag_s']['median'])}` s",
        f"- Response tau median: `{_fmt(response['tau_s']['median'])}` s",
        f"- Command/velocity gain median: `{_fmt(response['gain']['median'])}`",
        "",
        "## Calibrated Sensor Profile",
        "",
        "```json",
        json.dumps(profile, indent=2, sort_keys=True),
        "```",
        "",
        "## Per-Run Tracking",
        "",
        "| log | lag s | tau s | gain | rmse m/s | samples |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report["runs"]:
        tracking = run["tracking"]
        lines.append(
            f"| `{Path(run['log']).name}` | {_fmt(tracking['lag_s'])} | {_fmt(tracking['tau_s'])} | "
            f"{_fmt(tracking['gain'])} | {_fmt(tracking['rmse_m_s'])} | {tracking['samples']} |"
        )
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")


def _fmt(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4g}"
