from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from flightrl.hardware.avoidance_policy import reading_from_telemetry
from flightrl.hardware.target_conditioned_policy import TargetSpec, command_from_target_model, load_target_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a target-conditioned checkpoint on a directional Crazyflie log")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--target-direction-deg", type=float, required=True)
    parser.add_argument("--target-speed-m-s", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-speed-m-s", type=float, default=0.90)
    args = parser.parse_args()

    report = evaluate_log(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_report(report) + "\n")
    print(f"target_shadow_report={output}")
    print(f"samples={report['samples']} passed={report['passed']}")


def evaluate_log(args) -> dict:
    model = load_target_policy(args.checkpoint)
    target = TargetSpec(args.target_direction_deg, args.target_speed_m_s)
    actual = []
    predicted = []
    for row in csv.DictReader(Path(args.input).open()):
        if not has_required_columns(row):
            continue
        telemetry = {key: float(value) for key, value in row.items() if numeric(value)}
        command = command_from_target_model(model, reading_from_telemetry(telemetry), target, max_speed_m_s=args.max_speed_m_s)
        actual.append([float(row["vx_m_s"]), float(row["vy_m_s"]), float(row["yawrate_deg_s"]), float(row["zdistance_m"])])
        predicted.append([command.vx_m_s, command.vy_m_s, command.yawrate_deg_s, command.zdistance_m])
    if not actual:
        return {"checkpoint": args.checkpoint, "input": args.input, "samples": 0, "passed": False, "failures": ["no_command_rows"]}
    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)
    error = predicted_array - actual_array
    return {
        "checkpoint": args.checkpoint,
        "input": args.input,
        "target": {"direction_deg": args.target_direction_deg, "speed_m_s": args.target_speed_m_s},
        "samples": int(actual_array.shape[0]),
        "passed": True,
        "mae": {
            "vx_m_s": float(np.mean(np.abs(error[:, 0]))),
            "vy_m_s": float(np.mean(np.abs(error[:, 1]))),
            "yawrate_deg_s": float(np.mean(np.abs(error[:, 2]))),
            "zdistance_m": float(np.mean(np.abs(error[:, 3]))),
        },
        "direction_sign_agreement": {
            "vx": sign_agreement(actual_array[:, 0], predicted_array[:, 0]),
            "vy": sign_agreement(actual_array[:, 1], predicted_array[:, 1]),
        },
        "speed": {
            "actual_max_m_s": float(np.max(np.linalg.norm(actual_array[:, :2], axis=1))),
            "predicted_max_m_s": float(np.max(np.linalg.norm(predicted_array[:, :2], axis=1))),
        },
    }


def render_report(report: dict) -> str:
    if not report.get("passed"):
        return f"# Target-Conditioned Report\n\n- Passed: `False`\n- Failures: `{', '.join(report.get('failures', []))}`"
    mae = report["mae"]
    agreement = report["direction_sign_agreement"]
    return "\n".join(
        [
            "# Target-Conditioned Report",
            "",
            f"- Passed: `{report['passed']}`",
            f"- Samples: `{report['samples']}`",
            f"- MAE vx/vy/yaw/z: `{mae['vx_m_s']:.4f}`, `{mae['vy_m_s']:.4f}`, `{mae['yawrate_deg_s']:.4f}`, `{mae['zdistance_m']:.4f}`",
            f"- Direction sign agreement vx/vy: `{agreement['vx']:.3f}` / `{agreement['vy']:.3f}`",
        ]
    )


def has_required_columns(row: dict[str, str]) -> bool:
    return all(numeric(row.get(key, "")) for key in ("range.front", "range.back", "range.left", "range.right", "range.zrange", "vx_m_s", "vy_m_s", "yawrate_deg_s", "zdistance_m"))


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


if __name__ == "__main__":
    main()
