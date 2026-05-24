from __future__ import annotations

import csv
import json
from math import sqrt
from pathlib import Path
from typing import Any


DEFAULT_PAIRS = [
    ("vx_m_s", "stateEstimate.x"),
    ("vy_m_s", "stateEstimate.y"),
    ("yawrate_deg_s", "stabilizer.yaw"),
]


def summarize_latency(
    path: Path,
    *,
    pairs: list[tuple[str, str]] | None = None,
    max_lag_s: float = 0.5,
    min_abs_corr: float = 0.35,
    max_median_latency_s: float = 0.25,
) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open()))
    times = [value for row in rows if (value := _as_float(row.get("host_time_s"))) is not None]
    estimates = []
    for command, response in pairs or DEFAULT_PAIRS:
        estimate = estimate_pair(rows, command, response, max_lag_s=max_lag_s, min_abs_corr=min_abs_corr)
        if estimate["present"]:
            estimates.append(estimate)
    latencies = [estimate["latency_s"] for estimate in estimates if estimate.get("accepted")]
    failures = []
    if not estimates:
        failures.append("missing_pairs")
    if not latencies:
        failures.append("no_accepted_pairs")
    if latencies and median(latencies) > max_median_latency_s:
        failures.append("latency_slow")
    return {
        "input": str(path),
        "summary": {
            "rows": len(rows),
            "duration_s": max(times) - min(times) if len(times) >= 2 else 0.0,
            "latency_ready": not failures,
            "failures": failures,
            "accepted_pairs": len(latencies),
            "median_latency_s": median(latencies) if latencies else None,
            "max_lag_s": max_lag_s,
            "min_abs_corr": min_abs_corr,
            "max_median_latency_s": max_median_latency_s,
        },
        "pairs": estimates,
        "safety": "Latency evidence only; validate with replay gates before hardware policy deployment.",
    }


def estimate_pair(rows: list[dict[str, str]], command: str, response: str, *, max_lag_s: float, min_abs_corr: float = 0.35) -> dict[str, Any]:
    series = [(t, c, r) for row in rows if (t := _as_float(row.get("host_time_s"))) is not None and (c := _as_float(row.get(command))) is not None and (r := _as_float(row.get(response))) is not None]
    if len(series) < 5:
        return {"command": command, "response": response, "present": False}
    times = [row[0] for row in series]
    commands = [row[1] for row in series]
    responses = derivative(times, [row[2] for row in series])
    dt = median([b - a for a, b in zip(times, times[1:]) if b > a]) or 0.02
    max_lag_steps = max(0, int(round(max_lag_s / dt)))
    best = best_correlation(commands[1:], responses, max_lag_steps)
    return {
        "command": command,
        "response": response,
        "present": True,
        "samples": len(responses),
        "dt_s": dt,
        "latency_s": best["lag_steps"] * dt,
        "correlation": best["correlation"],
        "accepted": abs(best["correlation"]) >= min_abs_corr,
    }


def derivative(times: list[float], values: list[float]) -> list[float]:
    return [(value - prev_value) / (time - prev_time) for prev_time, time, prev_value, value in zip(times, times[1:], values, values[1:]) if time > prev_time]


def best_correlation(command: list[float], response: list[float], max_lag_steps: int) -> dict[str, float | int]:
    best = {"lag_steps": 0, "correlation": 0.0}
    for lag in range(max_lag_steps + 1):
        count = min(len(command) - lag, len(response))
        if count < 4:
            continue
        corr = correlation(command[:count], response[lag : lag + count])
        if abs(corr) > abs(best["correlation"]):
            best = {"lag_steps": lag, "correlation": corr}
    return best


def correlation(left: list[float], right: list[float]) -> float:
    mean_l = sum(left) / len(left)
    mean_r = sum(right) / len(right)
    num = sum((a - mean_l) * (b - mean_r) for a, b in zip(left, right))
    den_l = sqrt(sum((a - mean_l) ** 2 for a in left))
    den_r = sqrt(sum((b - mean_r) ** 2 for b in right))
    return num / (den_l * den_r) if den_l > 0 and den_r > 0 else 0.0


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Hardware Latency Summary",
        "",
        f"- Input: `{report['input']}`",
        f"- Ready: `{summary['latency_ready']}`",
        f"- Failures: `{', '.join(summary['failures']) or 'none'}`",
        f"- Median latency s: `{summary['median_latency_s']}`",
        "",
        "| command | response | accepted | latency s | correlation |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for pair in report["pairs"]:
        lines.append(f"| {pair['command']} | {pair['response']} | {pair['accepted']} | {_fmt(pair['latency_s'])} | {_fmt(pair['correlation'])} |")
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")


def _fmt(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.6g}"


def _as_float(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None
