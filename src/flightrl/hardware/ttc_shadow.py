from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from flightrl.hardware.avoidance_policy import reading_from_telemetry


GROUPS = ("all", "close_lt_18cm", "urgent_ttc_lt_35", "pinch_like")


def evaluate_ttc_shadow_log(
    input_csv: str | Path,
    *,
    target: str = "raw",
    shadow_prefix: str = "ttc_shadow",
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {name: [] for name in GROUPS}
    latest: dict[str, float] = {}
    for row in csv.DictReader(Path(input_csv).open()):
        latest.update({key: float(value) for key, value in row.items() if numeric(value)})
        if not has_command_pair(latest, target=target, shadow_prefix=shadow_prefix):
            continue
        actual = command_vector(latest, target)
        predicted = prefixed_command_vector(latest, shadow_prefix)
        for group in groups_for_row(latest):
            grouped[group].append((actual, predicted))
    return {
        "input": str(input_csv),
        "target": target,
        "shadow_prefix": shadow_prefix,
        "groups": {name: group_metrics(pairs) for name, pairs in grouped.items()},
    }


def write_ttc_shadow_report(report: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    path.with_suffix(".md").write_text(render_ttc_shadow_markdown(report) + "\n")


def render_ttc_shadow_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TTC Shadow Gap Report",
        "",
        f"- Input: `{report['input']}`",
        f"- Target command: `{report['target']}`",
        f"- Shadow prefix: `{report['shadow_prefix']}`",
        "",
        "| group | samples | MAE vx/vy/z | sign vx/vy | speed p95 | L2 p95 |",
        "| --- | ---: | --- | --- | ---: | ---: |",
    ]
    for name, metrics in report["groups"].items():
        if metrics["samples"] == 0:
            lines.append(f"| `{name}` | 0 | n/a | n/a | n/a | n/a |")
            continue
        mae = metrics["mae"]
        sign = metrics["direction_sign_agreement"]
        lines.append(
            f"| `{name}` | {metrics['samples']} | "
            f"{mae['vx_m_s']:.4f} / {mae['vy_m_s']:.4f} / {mae['zdistance_m']:.4f} | "
            f"{sign['vx']:.3f} / {sign['vy']:.3f} | "
            f"{metrics['speed']['shadow_p95_m_s']:.3f} | {metrics['l2_error_p95']:.4f} |"
        )
    return "\n".join(lines)


def groups_for_row(telemetry: dict[str, float]) -> list[str]:
    groups = ["all"]
    min_range = float(telemetry.get("min_horizontal_range_m", 4.0))
    min_ttc = float(telemetry.get("min_horizontal_ttc_s", 99.0))
    if min_range < 0.18:
        groups.append("close_lt_18cm")
    if min_ttc < 0.35:
        groups.append("urgent_ttc_lt_35")
    if is_pinch_like(telemetry):
        groups.append("pinch_like")
    return groups


def is_pinch_like(telemetry: dict[str, float]) -> bool:
    reading = reading_from_telemetry(telemetry)
    front_back = reading.front_m < 0.35 and reading.back_m < 0.35 and max(reading.left_m, reading.right_m) > 0.8
    left_right = reading.left_m < 0.35 and reading.right_m < 0.35 and max(reading.front_m, reading.back_m) > 0.8
    return front_back or left_right


def group_metrics(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    if not pairs:
        return {"samples": 0}
    actual = np.stack([pair[0] for pair in pairs])
    predicted = np.stack([pair[1] for pair in pairs])
    error = predicted - actual
    speed = np.linalg.norm(predicted[:, :2], axis=1)
    l2_error = np.linalg.norm(error[:, :2], axis=1)
    return {
        "samples": int(actual.shape[0]),
        "mae": {
            "vx_m_s": float(np.mean(np.abs(error[:, 0]))),
            "vy_m_s": float(np.mean(np.abs(error[:, 1]))),
            "yawrate_deg_s": float(np.mean(np.abs(error[:, 2]))),
            "zdistance_m": float(np.mean(np.abs(error[:, 3]))),
        },
        "direction_sign_agreement": {
            "vx": sign_agreement(actual[:, 0], predicted[:, 0]),
            "vy": sign_agreement(actual[:, 1], predicted[:, 1]),
        },
        "speed": {
            "shadow_p95_m_s": float(np.percentile(speed, 95)),
            "shadow_max_m_s": float(np.max(speed)),
        },
        "l2_error_p95": float(np.percentile(l2_error, 95)),
    }


def has_command_pair(row: dict[str, str], *, target: str, shadow_prefix: str) -> bool:
    return all(numeric(row.get(key, "")) for key in (*command_keys(target), *prefixed_command_keys(shadow_prefix)))


def command_vector(row: dict[str, str], target: str) -> np.ndarray:
    return np.asarray([float(row[key]) for key in command_keys(target)], dtype=np.float64)


def prefixed_command_vector(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray([float(row[key]) for key in prefixed_command_keys(prefix)], dtype=np.float64)


def command_keys(target: str) -> tuple[str, str, str, str]:
    if target == "raw":
        return ("raw_vx_m_s", "raw_vy_m_s", "raw_yawrate_deg_s", "raw_zdistance_m")
    if target == "held":
        return ("vx_m_s", "vy_m_s", "yawrate_deg_s", "zdistance_m")
    raise ValueError("target must be 'raw' or 'held'")


def prefixed_command_keys(prefix: str) -> tuple[str, str, str, str]:
    return (
        f"{prefix}_vx_m_s",
        f"{prefix}_vy_m_s",
        f"{prefix}_yawrate_deg_s",
        f"{prefix}_zdistance_m",
    )


def numeric(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def sign_agreement(actual: np.ndarray, predicted: np.ndarray, *, threshold: float = 0.05) -> float:
    mask = np.abs(actual) >= threshold
    if not np.any(mask):
        return 1.0
    return float(np.mean(np.sign(actual[mask]) == np.sign(predicted[mask])))
