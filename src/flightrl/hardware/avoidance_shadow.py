from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from flightrl.hardware.avoidance_policy import (
    AvoidanceCommand,
    RangerAvoidancePolicy,
    command_from_model,
    reading_from_telemetry,
)


def load_ranger_policy(path: str | Path) -> RangerAvoidancePolicy:
    checkpoint = torch.load(path, map_location="cpu")
    hidden_size = int(checkpoint.get("hidden_size", 64))
    model = RangerAvoidancePolicy(hidden_size=hidden_size)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def shadow_command_row(command: AvoidanceCommand, *, prefix: str = "shadow") -> dict[str, float]:
    return {
        f"{prefix}_vx_m_s": command.vx_m_s,
        f"{prefix}_vy_m_s": command.vy_m_s,
        f"{prefix}_yawrate_deg_s": command.yawrate_deg_s,
        f"{prefix}_zdistance_m": command.zdistance_m,
    }


def evaluate_shadow_log(
    *,
    checkpoint: str | Path,
    input_csv: str | Path,
    max_speed_m_s: float,
    max_yawrate_deg_s: float = 45.0,
) -> dict[str, Any]:
    model = load_ranger_policy(checkpoint)
    rows = list(csv.DictReader(Path(input_csv).open()))
    pairs = []
    for row in rows:
        if not has_actual_command(row):
            continue
        telemetry = {key: float(value) for key, value in row.items() if numeric(value)}
        shadow = command_from_model(
            model,
            reading_from_telemetry(telemetry),
            max_speed_m_s=max_speed_m_s,
            max_yawrate_deg_s=max_yawrate_deg_s,
        )
        actual = np.asarray([float(row["vx_m_s"]), float(row["vy_m_s"]), float(row["yawrate_deg_s"]), float(row["zdistance_m"])])
        predicted = np.asarray([shadow.vx_m_s, shadow.vy_m_s, shadow.yawrate_deg_s, shadow.zdistance_m])
        pairs.append((actual, predicted))
    return shadow_report(checkpoint, input_csv, pairs)


def write_shadow_report(report: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    path.with_suffix(".md").write_text(render_shadow_markdown(report) + "\n")


def shadow_report(checkpoint: str | Path, input_csv: str | Path, pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    if not pairs:
        return {"checkpoint": str(checkpoint), "input": str(input_csv), "samples": 0, "passed": False, "failures": ["no_command_rows"]}
    actual = np.stack([pair[0] for pair in pairs])
    predicted = np.stack([pair[1] for pair in pairs])
    error = predicted - actual
    speed_actual = np.linalg.norm(actual[:, :2], axis=1)
    speed_predicted = np.linalg.norm(predicted[:, :2], axis=1)
    return {
        "checkpoint": str(checkpoint),
        "input": str(input_csv),
        "samples": int(actual.shape[0]),
        "passed": True,
        "mae": {
            "vx_m_s": float(np.mean(np.abs(error[:, 0]))),
            "vy_m_s": float(np.mean(np.abs(error[:, 1]))),
            "yawrate_deg_s": float(np.mean(np.abs(error[:, 2]))),
            "zdistance_m": float(np.mean(np.abs(error[:, 3]))),
        },
        "speed": {
            "actual_max_m_s": float(np.max(speed_actual)),
            "shadow_max_m_s": float(np.max(speed_predicted)),
            "shadow_over_actual_max_ratio": float(np.max(speed_predicted) / max(np.max(speed_actual), 1e-6)),
        },
        "direction_sign_agreement": {
            "vx": sign_agreement(actual[:, 0], predicted[:, 0]),
            "vy": sign_agreement(actual[:, 1], predicted[:, 1]),
        },
    }


def render_shadow_markdown(report: dict[str, Any]) -> str:
    if not report.get("passed"):
        return f"# Ranger Shadow Report\n\n- Passed: `False`\n- Failures: `{', '.join(report.get('failures', []))}`"
    mae = report["mae"]
    speed = report["speed"]
    agreement = report["direction_sign_agreement"]
    return "\n".join(
        [
            "# Ranger Shadow Report",
            "",
            f"- Passed: `{report['passed']}`",
            f"- Samples: `{report['samples']}`",
            f"- Checkpoint: `{report['checkpoint']}`",
            f"- Input: `{report['input']}`",
            f"- MAE vx/vy/yaw/z: `{mae['vx_m_s']:.4f}`, `{mae['vy_m_s']:.4f}`, `{mae['yawrate_deg_s']:.4f}`, `{mae['zdistance_m']:.4f}`",
            f"- Max speed actual/shadow: `{speed['actual_max_m_s']:.3f}` / `{speed['shadow_max_m_s']:.3f}` m/s",
            f"- Direction sign agreement vx/vy: `{agreement['vx']:.3f}` / `{agreement['vy']:.3f}`",
        ]
    )


def has_actual_command(row: dict[str, str]) -> bool:
    return all(numeric(row.get(key, "")) for key in ("vx_m_s", "vy_m_s", "yawrate_deg_s", "zdistance_m"))


def numeric(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def sign_agreement(actual: np.ndarray, predicted: np.ndarray, *, threshold: float = 0.03) -> float:
    mask = np.abs(actual) >= threshold
    if not np.any(mask):
        return 1.0
    return float(np.mean(np.sign(actual[mask]) == np.sign(predicted[mask])))
