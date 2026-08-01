from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from flightrl.semantic.puffer_shadow import SemanticPufferShadow


def replay_shadow_run(
    run_dir: Path,
    checkpoint: Path,
    training_report: Path,
    *,
    suppress_detections: bool = False,
    assumed_target_distance_m: float = 2.0,
) -> list[dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    shadow = SemanticPufferShadow.from_training_report(
        checkpoint,
        training_report,
        assumed_target_distance_m=assumed_target_distance_m,
    )
    rows = []
    for line in (run_dir / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        grounding = event["grounding"]
        detections = grounding.get("detections", [])
        best = max(detections, key=lambda item: item["confidence"], default=None)
        if suppress_detections:
            best = None
        frame_path = Path(event["frame_path"])
        if not frame_path.is_absolute():
            frame_path = Path.cwd() / frame_path
        frame = np.asarray(Image.open(frame_path).convert("L"))
        prediction = shadow.step(
            frame=frame,
            telemetry=event.get("telemetry", {}),
            prompt=str(manifest["prompt"]),
            detection=best,
        )
        rows.append(
            {
                "frame_index": grounding["frame_index"],
                "frame_host_time_s": grounding["frame_host_time_s"],
                "detection_confidence": None if best is None else best["confidence"],
                "detection_horizontal_error": _horizontal_error(best),
                **prediction,
            }
        )
    return rows


def semantic_shadow_gate(
    rows: list[dict],
    suppressed_rows: list[dict],
) -> dict[str, float | bool | int]:
    numeric_actions = (
        "vx_body_m_s",
        "vy_body_m_s",
        "vz_m_s",
        "yawrate_deg_s",
    )
    finite = all(
        np.isfinite(float(row[key]))
        for row in (*rows, *suppressed_rows)
        for key in numeric_actions
    )
    preacquisition = [row for row in rows if not row["target_acquired"]]
    detections = [row for row in rows if row["target_detected"]]
    directional = [
        row
        for row in detections
        if abs(float(row["detection_horizontal_error"])) >= 0.08
    ]
    direction_correct = [
        np.sign(float(row["yawrate_deg_s"]))
        == np.sign(-float(row["detection_horizontal_error"]))
        for row in directional
    ]
    metrics: dict[str, float | bool | int] = {
        "processed_frames": len(rows),
        "frames_with_detection": len(detections),
        "all_actions_finite": finite,
        "controls_drone": False,
        "preacquisition_horizontal_p95_m_s": _horizontal_percentile(
            preacquisition,
            95,
        ),
        "suppressed_detection_horizontal_p95_m_s": _horizontal_percentile(
            suppressed_rows,
            95,
        ),
        "max_horizontal_speed_m_s": _horizontal_percentile(rows, 100),
        "max_abs_vertical_speed_m_s": _absolute_percentile(rows, "vz_m_s", 100),
        "max_abs_yawrate_deg_s": _absolute_percentile(
            rows,
            "yawrate_deg_s",
            100,
        ),
        "detection_abs_yawrate_p95_deg_s": _absolute_percentile(
            detections,
            "yawrate_deg_s",
            95,
        ),
        "detection_yaw_sign_accuracy": float(np.mean(direction_correct))
        if direction_correct
        else 0.0,
        "directional_detection_frames": len(directional),
    }
    metrics["next_live_shadow_gate_passed"] = bool(
        finite
        and len(detections) >= 5
        and metrics["preacquisition_horizontal_p95_m_s"] <= 0.03
        and metrics["suppressed_detection_horizontal_p95_m_s"] <= 0.03
        and metrics["max_horizontal_speed_m_s"] <= 0.101
        and metrics["max_abs_vertical_speed_m_s"] <= 0.101
        and metrics["max_abs_yawrate_deg_s"] <= 20.1
        and metrics["detection_abs_yawrate_p95_deg_s"] <= 10.1
        and metrics["detection_yaw_sign_accuracy"] >= 0.90
    )
    return metrics


def semantic_translation_shadow_gate(
    rows: list[dict],
) -> dict[str, float | bool | int]:
    numeric = (
        "vx_body_m_s",
        "vy_body_m_s",
        "vz_m_s",
        "yawrate_deg_s",
        "predicted_clearance_m",
        "predicted_collision_risk",
    )
    finite = bool(rows) and all(
        row[key] is not None and np.isfinite(float(row[key]))
        for row in rows
        for key in numeric
    )
    unsafe_forward = [
        row
        for row in rows
        if float(row["vx_body_m_s"]) > 0.01
        and (
            float(row["predicted_clearance_m"]) < 0.45
            or float(row["predicted_collision_risk"]) > 0.50
        )
    ]
    metrics: dict[str, float | bool | int] = {
        "translation_processed_frames": len(rows),
        "translation_frames_with_detection": sum(
            bool(row["target_detected"]) for row in rows
        ),
        "translation_all_actions_and_safety_finite": finite,
        "translation_controls_drone": any(
            bool(row["controls_drone"]) for row in rows
        ),
        "translation_max_forward_speed_m_s": _absolute_percentile(
            rows, "vx_body_m_s", 100
        ),
        "translation_max_lateral_vertical_speed_m_s": max(
            _absolute_percentile(rows, "vy_body_m_s", 100),
            _absolute_percentile(rows, "vz_m_s", 100),
        ),
        "translation_max_abs_yawrate_deg_s": _absolute_percentile(
            rows, "yawrate_deg_s", 100
        ),
        "translation_min_predicted_clearance_m": _percentile(
            rows, "predicted_clearance_m", 0
        ),
        "translation_max_predicted_collision_risk": _percentile(
            rows, "predicted_collision_risk", 100
        ),
        "translation_unsafe_forward_fraction": len(unsafe_forward)
        / max(1, len(rows)),
    }
    metrics["translation_shadow_gate_passed"] = bool(
        finite
        and len(rows) >= 20
        and metrics["translation_frames_with_detection"] >= 5
        and metrics["translation_controls_drone"] is False
        and metrics["translation_max_forward_speed_m_s"] <= 0.151
        and metrics["translation_max_lateral_vertical_speed_m_s"] <= 0.001
        and metrics["translation_max_abs_yawrate_deg_s"] <= 20.1
        and metrics["translation_min_predicted_clearance_m"] >= 0.0
        and metrics["translation_max_predicted_collision_risk"] <= 1.0
        and metrics["translation_unsafe_forward_fraction"] == 0.0
    )
    return metrics


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["frame_index"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _horizontal_error(detection: dict | None) -> float:
    if detection is None:
        return 0.0
    box = detection["box"]
    return 0.5 * (float(box["x_min"]) + float(box["x_max"])) - 0.5


def _horizontal_percentile(rows: list[dict], percentile: float) -> float:
    speeds = [float(np.hypot(row["vx_body_m_s"], row["vy_body_m_s"])) for row in rows]
    return float(np.percentile(speeds, percentile)) if speeds else 0.0


def _absolute_percentile(
    rows: list[dict],
    key: str,
    percentile: float,
) -> float:
    values = [abs(float(row[key])) for row in rows]
    return float(np.percentile(values, percentile)) if values else 0.0


def _percentile(rows: list[dict], key: str, percentile: float) -> float:
    values = [float(row[key]) for row in rows if row[key] is not None]
    return float(np.percentile(values, percentile)) if values else float("nan")
