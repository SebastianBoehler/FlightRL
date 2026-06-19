from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flightrl.hardware.avoidance_policy import AvoidanceCommand, reactive_clearance_command, reading_from_telemetry
from flightrl.hardware.target_direction import TargetDirectionConfig, cruise_vector, target_direction_command


@dataclass(frozen=True, slots=True)
class LogSpec:
    path: Path
    direction_deg: float
    speed_m_s: float


@dataclass(frozen=True, slots=True)
class Baseline:
    name: str
    config: TargetDirectionConfig | None = None
    reactive_clearance_m: float | None = None
    reactive_speed_m_s: float = 0.25
    cruise_only: bool = False


BASELINES = (
    Baseline("hover_zero"),
    Baseline("cruise_only", cruise_only=True),
    Baseline("reactive_045_slow", reactive_clearance_m=0.45, reactive_speed_m_s=0.25),
    Baseline("reactive_075_fast", reactive_clearance_m=0.75, reactive_speed_m_s=0.75),
    Baseline("target_dir_close_gentle", config=TargetDirectionConfig(clearance_m=0.55, avoidance_speed_m_s=0.35, max_speed_m_s=0.35)),
    Baseline("target_dir_close_snappy", config=TargetDirectionConfig(clearance_m=0.75, avoidance_speed_m_s=0.80, max_speed_m_s=0.80)),
    Baseline("target_dir_current_fast", config=TargetDirectionConfig(clearance_m=1.30, avoidance_speed_m_s=0.95, max_speed_m_s=0.95)),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare deterministic ranger baselines against Crazyflie command logs.")
    parser.add_argument("--input", action="append", required=True, metavar="CSV:DIRECTION_DEG:SPEED_M_S")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-height-m", type=float, default=0.50)
    args = parser.parse_args()

    specs = [parse_input(value) for value in args.input]
    rows = [entry for spec in specs for entry in load_log(spec)]
    if not rows:
        raise SystemExit("no usable command rows found")
    report = build_report(rows, args.target_height_m)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_report(report) + "\n")
    print(f"deterministic_baseline_report={output}")
    print(f"rows={report['rows']} best_mae={report['best_by_vxy_mae']['name']}")


def parse_input(value: str) -> LogSpec:
    parts = value.rsplit(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--input must be CSV:DIRECTION_DEG:SPEED_M_S")
    return LogSpec(Path(parts[0]), float(parts[1]), float(parts[2]))


def load_log(spec: LogSpec) -> list[dict]:
    entries = []
    for row in csv.DictReader(spec.path.open()):
        if not has_required_columns(row):
            continue
        telemetry = {key: float(value) for key, value in row.items() if numeric(value)}
        reading = reading_from_telemetry(telemetry)
        entries.append(
            {
                "log": spec.path.name,
                "direction_deg": spec.direction_deg,
                "speed_m_s": spec.speed_m_s,
                "reading": reading,
                "actual": np.asarray(
                    [
                        float(row["vx_m_s"]),
                        float(row["vy_m_s"]),
                        float(row["yawrate_deg_s"]),
                        float(row["zdistance_m"]),
                    ],
                    dtype=np.float32,
                ),
            }
        )
    return entries


def build_report(rows: list[dict], target_height_m: float) -> dict:
    results = [score_baseline(baseline, rows, target_height_m) for baseline in BASELINES]
    return {
        "rows": len(rows),
        "logs": sorted({row["log"] for row in rows}),
        "baselines": results,
        "best_by_vxy_mae": min(results, key=lambda item: item["mae"]["vxy_m_s"]),
        "best_by_close_escape": max(results, key=lambda item: item["close_escape_agreement"]),
    }


def score_baseline(baseline: Baseline, rows: list[dict], target_height_m: float) -> dict:
    actual = np.stack([row["actual"] for row in rows])
    predicted = np.stack([command_array(predict(baseline, row, target_height_m)) for row in rows])
    error = predicted - actual
    speed = np.linalg.norm(predicted[:, :2], axis=1)
    return {
        "name": baseline.name,
        "mae": {
            "vx_m_s": float(np.mean(np.abs(error[:, 0]))),
            "vy_m_s": float(np.mean(np.abs(error[:, 1]))),
            "vxy_m_s": float(np.mean(np.linalg.norm(error[:, :2], axis=1))),
            "yawrate_deg_s": float(np.mean(np.abs(error[:, 2]))),
            "zdistance_m": float(np.mean(np.abs(error[:, 3]))),
        },
        "sign_agreement": {
            "vx": sign_agreement(actual[:, 0], predicted[:, 0]),
            "vy": sign_agreement(actual[:, 1], predicted[:, 1]),
        },
        "speed": {
            "mean_m_s": float(np.mean(speed)),
            "p95_m_s": float(np.quantile(speed, 0.95)),
            "max_m_s": float(np.max(speed)),
        },
        "smoothness": {"mean_step_m_s": mean_step(predicted[:, :2])},
        "close_escape_agreement": closest_side_escape_agreement(rows, predicted),
    }


def predict(baseline: Baseline, row: dict, target_height_m: float) -> AvoidanceCommand:
    reading = row["reading"]
    if baseline.cruise_only:
        vx, vy = cruise_vector(row["direction_deg"], row["speed_m_s"])
        return AvoidanceCommand(vx, vy, 0.0, target_height_m)
    if baseline.reactive_clearance_m is not None:
        return reactive_clearance_command(
            reading,
            clearance_m=baseline.reactive_clearance_m,
            hard_clearance_m=0.10,
            target_height_m=target_height_m,
            max_speed_m_s=baseline.reactive_speed_m_s,
        )
    if baseline.config is not None:
        config = TargetDirectionConfig(
            direction_deg=row["direction_deg"],
            target_speed_m_s=row["speed_m_s"],
            clearance_m=baseline.config.clearance_m,
            hard_clearance_m=baseline.config.hard_clearance_m,
            target_height_m=target_height_m,
            avoidance_speed_m_s=baseline.config.avoidance_speed_m_s,
            max_speed_m_s=baseline.config.max_speed_m_s,
            slowdown_gain=baseline.config.slowdown_gain,
            avoidance_gain=baseline.config.avoidance_gain,
        )
        return target_direction_command(reading, config)
    return AvoidanceCommand(0.0, 0.0, 0.0, target_height_m)


def command_array(command: AvoidanceCommand) -> np.ndarray:
    return np.asarray([command.vx_m_s, command.vy_m_s, command.yawrate_deg_s, command.zdistance_m], dtype=np.float32)


def closest_side_escape_agreement(rows: list[dict], predicted: np.ndarray, threshold_m: float = 0.35) -> float:
    expected = []
    observed = []
    for row, command in zip(rows, predicted, strict=True):
        reading = row["reading"]
        distances = np.asarray([reading.front_m, reading.back_m, reading.left_m, reading.right_m])
        side = int(np.argmin(distances))
        if distances[side] >= threshold_m:
            continue
        if side == 0:
            expected.append(-1.0)
            observed.append(command[0])
        elif side == 1:
            expected.append(1.0)
            observed.append(command[0])
        elif side == 2:
            expected.append(-1.0)
            observed.append(command[1])
        else:
            expected.append(1.0)
            observed.append(command[1])
    if not expected:
        return 1.0
    expected_array = np.asarray(expected)
    observed_array = np.asarray(observed)
    return float(np.mean(np.sign(expected_array) == np.sign(observed_array)))


def sign_agreement(actual: np.ndarray, predicted: np.ndarray, threshold: float = 0.03) -> float:
    mask = np.abs(actual) >= threshold
    if not np.any(mask):
        return 1.0
    return float(np.mean(np.sign(actual[mask]) == np.sign(predicted[mask])))


def mean_step(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(values, axis=0), axis=1)))


def has_required_columns(row: dict[str, str]) -> bool:
    keys = ("range.front", "range.back", "range.left", "range.right", "range.zrange", "vx_m_s", "vy_m_s", "yawrate_deg_s", "zdistance_m")
    return all(numeric(row.get(key, "")) for key in keys)


def numeric(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def render_report(report: dict) -> str:
    lines = [
        "# Deterministic Avoidance Baselines",
        "",
        f"- Rows: `{report['rows']}`",
        f"- Logs: `{len(report['logs'])}`",
        f"- Best vxy MAE: `{report['best_by_vxy_mae']['name']}`",
        f"- Best close escape: `{report['best_by_close_escape']['name']}`",
        "",
        "| baseline | vxy MAE m/s | vx sign | vy sign | close escape | speed p95 m/s | step mean m/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["baselines"]:
        lines.append(
            f"| {item['name']} | {item['mae']['vxy_m_s']:.4f} | {item['sign_agreement']['vx']:.3f} | "
            f"{item['sign_agreement']['vy']:.3f} | {item['close_escape_agreement']:.3f} | "
            f"{item['speed']['p95_m_s']:.3f} | {item['smoothness']['mean_step_m_s']:.4f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
