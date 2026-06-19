from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


STATE_KEYS = ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z")
ATTITUDE_KEYS = ("stabilizer.roll", "stabilizer.pitch", "stabilizer.yaw")
SIDE_RANGE_KEYS = ("range.front", "range.back", "range.left", "range.right")


def summarize_hover_yaw_logs(
    logs: list[Path],
    *,
    contaminated_logs: list[Path] | None = None,
    stable_after_s: float = 1.0,
    max_xy_span_m: float = 0.18,
    max_z_span_m: float = 0.08,
    min_side_range_mm: float = 450.0,
    min_battery_v: float = 3.7,
    max_commanded_xy_m_s: float = 1e-6,
) -> dict[str, Any]:
    clean = [
        summarize_log(
            path,
            stable_after_s=stable_after_s,
            max_xy_span_m=max_xy_span_m,
            max_z_span_m=max_z_span_m,
            min_side_range_mm=min_side_range_mm,
            min_battery_v=min_battery_v,
            max_commanded_xy_m_s=max_commanded_xy_m_s,
        )
        for path in logs
    ]
    contaminated = [summarize_contaminated(path) for path in contaminated_logs or []]
    failures = [record["path"] for record in clean if not record["passed"]]
    report = {
        "summary": {
            "stability_ready": bool(clean) and not failures,
            "clean_logs": len(clean),
            "passing_clean_logs": len(clean) - len(failures),
            "contaminated_logs": len(contaminated),
            "failed_clean_logs": failures,
            "max_stable_xy_span_m": max_or_none(record["stable"]["xy_span_m"] for record in clean),
            "min_stable_side_range_mm": min_or_none(record["stable"]["min_side_range_mm"] for record in clean),
            "min_battery_v": min_or_none(record["battery"]["min_v"] for record in clean),
        },
        "thresholds": {
            "stable_after_s": stable_after_s,
            "max_xy_span_m": max_xy_span_m,
            "max_z_span_m": max_z_span_m,
            "min_side_range_mm": min_side_range_mm,
            "min_battery_v": min_battery_v,
            "max_commanded_xy_m_s": max_commanded_xy_m_s,
        },
        "clean_logs": clean,
        "contaminated_logs": contaminated,
        "safety": "Firmware hover/yaw stability evidence only; this does not approve learned-policy flight.",
    }
    return report


def summarize_log(path: Path, **thresholds: float) -> dict[str, Any]:
    rows = read_rows(path)
    stable = stable_rows(rows, thresholds["stable_after_s"])
    command = command_summary(rows)
    stable_metrics = motion_summary(stable)
    battery = signal_summary(rows, "pm.vbat")
    failures = failures_for(command, stable_metrics, battery, thresholds)
    return {
        "path": str(path),
        "passed": not failures,
        "failures": failures,
        "rows": len(rows),
        "duration_s": duration_s(rows),
        "modes": modes(rows),
        "command": command,
        "overall": motion_summary(rows),
        "stable": stable_metrics,
        "battery": battery,
    }


def summarize_contaminated(path: Path) -> dict[str, Any]:
    rows = read_rows(path)
    return {
        "path": str(path),
        "rows": len(rows),
        "duration_s": duration_s(rows),
        "modes": modes(rows),
        "overall": motion_summary(rows),
        "battery": signal_summary(rows, "pm.vbat"),
        "classification": "contaminated_negative",
    }


def failures_for(command: dict[str, float], stable: dict[str, Any], battery: dict[str, float | None], thresholds: dict[str, float]) -> list[str]:
    failures = []
    if command["max_abs_vx_m_s"] > thresholds["max_commanded_xy_m_s"] or command["max_abs_vy_m_s"] > thresholds["max_commanded_xy_m_s"]:
        failures.append("xy_commanded")
    if stable["xy_span_m"] is None or stable["xy_span_m"] > thresholds["max_xy_span_m"]:
        failures.append("xy_drift")
    if stable["z_span_m"] is None or stable["z_span_m"] > thresholds["max_z_span_m"]:
        failures.append("z_hold")
    if stable["min_side_range_mm"] is None or stable["min_side_range_mm"] < thresholds["min_side_range_mm"]:
        failures.append("clearance")
    if battery["min_v"] is None or battery["min_v"] < thresholds["min_battery_v"]:
        failures.append("battery")
    return failures


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def stable_rows(rows: list[dict[str, str]], stable_after_s: float) -> list[dict[str, str]]:
    if not rows:
        return []
    start = value(rows[0], "host_time_s")
    if start is None:
        return rows
    stable = [row for row in rows if (value(row, "host_time_s") or start) - start >= stable_after_s]
    return stable or rows


def command_summary(rows: list[dict[str, str]]) -> dict[str, float]:
    return {
        "max_abs_vx_m_s": max_abs(rows, "vx_m_s"),
        "max_abs_vy_m_s": max_abs(rows, "vy_m_s"),
        "max_abs_yawrate_deg_s": max_abs(rows, "yawrate_deg_s"),
    }


def motion_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    x = values(rows, "stateEstimate.x")
    y = values(rows, "stateEstimate.y")
    z = values(rows, "stateEstimate.z")
    side_ranges = [v for row in rows for key in SIDE_RANGE_KEYS if (v := value(row, key)) is not None and v < 32766.0]
    return {
        "xy_span_m": xy_span(x, y),
        "x_span_m": span(x),
        "y_span_m": span(y),
        "z_span_m": span(z),
        "yaw_span_deg": span(values(rows, "stabilizer.yaw")),
        "min_side_range_mm": min_or_none(side_ranges),
        "range_z_summary": signal_summary(rows, "range.zrange"),
        "state_summary": {key: signal_summary(rows, key) for key in STATE_KEYS},
        "attitude_summary": {key: signal_summary(rows, key) for key in ATTITUDE_KEYS},
    }


def signal_summary(rows: list[dict[str, str]], key: str) -> dict[str, float | None]:
    vals = values(rows, key)
    return {
        "min_v": min_or_none(vals),
        "max_v": max_or_none(vals),
        "mean_v": mean(vals) if vals else None,
        "std_v": pstdev(vals) if len(vals) > 1 else (0.0 if vals else None),
        "span_v": span(vals),
    }


def values(rows: list[dict[str, str]], key: str) -> list[float]:
    vals = []
    for row in rows:
        parsed = value(row, key)
        if parsed is not None:
            vals.append(parsed)
    return vals


def value(row: dict[str, str], key: str) -> float | None:
    try:
        parsed = float(row.get(key, ""))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def max_abs(rows: list[dict[str, str]], key: str) -> float:
    vals = values(rows, key)
    return max((abs(v) for v in vals), default=0.0)


def xy_span(x: list[float], y: list[float]) -> float | None:
    if not x or not y:
        return None
    return math.hypot(span(x) or 0.0, span(y) or 0.0)


def span(vals: list[float]) -> float | None:
    return max(vals) - min(vals) if vals else None


def duration_s(rows: list[dict[str, str]]) -> float | None:
    times = values(rows, "host_time_s")
    return span(times)


def modes(rows: list[dict[str, str]]) -> list[str]:
    seen = []
    for row in rows:
        mode = row.get("mode", "")
        if mode and mode not in seen:
            seen.append(mode)
    return seen


def min_or_none(vals) -> float | None:
    clean = [v for v in vals if v is not None]
    return min(clean) if clean else None


def max_or_none(vals) -> float | None:
    clean = [v for v in vals if v is not None]
    return max(clean) if clean else None


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Firmware Hover/Yaw Stability",
        "",
        f"- Stability ready: `{summary['stability_ready']}`",
        f"- Clean logs: `{summary['passing_clean_logs']}/{summary['clean_logs']}`",
        f"- Contaminated negative logs: `{summary['contaminated_logs']}`",
        f"- Max stable XY span: `{summary['max_stable_xy_span_m']}` m",
        f"- Min stable side range: `{summary['min_stable_side_range_mm']}` mm",
        f"- Min battery: `{summary['min_battery_v']}` V",
        "",
        "| log | passed | failures | yaw cmd deg/s | stable xy m | stable z m | yaw span deg | min side mm | min battery V |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in report["clean_logs"]:
        lines.append(
            f"| `{Path(record['path']).name}` | {record['passed']} | {', '.join(record['failures']) or 'none'} | "
            f"{fmt(record['command']['max_abs_yawrate_deg_s'])} | {fmt(record['stable']['xy_span_m'])} | "
            f"{fmt(record['stable']['z_span_m'])} | {fmt(record['stable']['yaw_span_deg'])} | "
            f"{fmt(record['stable']['min_side_range_mm'])} | {fmt(record['battery']['min_v'])} |"
        )
    if report["contaminated_logs"]:
        lines.extend(["", "## Contaminated Negative Logs", ""])
        for record in report["contaminated_logs"]:
            lines.append(f"- `{Path(record['path']).name}`: {record['classification']}")
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def fmt(value_: float | None) -> str:
    return "n/a" if value_ is None else f"{value_:.4g}"
