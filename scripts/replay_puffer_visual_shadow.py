from __future__ import annotations

import argparse
import csv
import json
from math import cos, radians, sin
from pathlib import Path

import numpy as np
from PIL import Image

from flightrl.puffer4_vision_runtime import VisualPufferShadow
from flightrl.puffer4_vision_sections import (
    NAVIGATION_RESIDUAL_SCALE,
)


SHADOW_WAYPOINT_SPEED_M_S = 0.08


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay AI Deck frames through a visual Puffer checkpoint"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-distance-m", type=float, default=3.1)
    args = parser.parse_args()

    events = [
        json.loads(line)
        for line in (args.run_dir / "events.jsonl").read_text().splitlines()
    ]
    if not events:
        raise SystemExit("run has no events")
    target = _target_from_first_event(events[0], args.target_distance_m)
    shadow = VisualPufferShadow(args.checkpoint)
    rows = [_replay_event(shadow, event, target) for event in events]
    output = args.output or args.run_dir / "puffer_visual_shadow.csv"
    _write_rows(output, rows)
    summary = _summary(args.checkpoint, rows, target)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"shadow_rows={len(rows)}")
    print("controls_drone=False")
    print(f"output={output}")
    print(f"summary={summary_path}")


def _target_from_first_event(event: dict, distance_m: float) -> tuple[float, ...]:
    telemetry = event.get("telemetry", {})
    x = float(telemetry.get("stateEstimate.x", 0.0))
    y = float(telemetry.get("stateEstimate.y", 0.0))
    z = float(telemetry.get("stateEstimate.z", 0.0))
    yaw = radians(float(telemetry.get("stateEstimate.yaw", 0.0)))
    return (
        x + distance_m * cos(yaw),
        y + distance_m * sin(yaw),
        z,
        yaw,
    )


def _replay_event(
    shadow: VisualPufferShadow,
    event: dict,
    target: tuple[float, ...],
) -> dict:
    frame_path = Path(event["frame_path"])
    if not frame_path.is_absolute():
        frame_path = Path.cwd() / frame_path
    frame = np.asarray(Image.open(frame_path).convert("L"))
    intent = _goal_intent(event.get("telemetry", {}), target)
    prediction = shadow.step(frame, intent)
    action_vy = float(prediction["action_vy"])
    return {
        "frame_index": event["grounding"]["frame_index"],
        "frame_mean": float(frame.mean()),
        "intent_distance": float(intent[3]),
        "residual_vy_m_s": (
            NAVIGATION_RESIDUAL_SCALE * SHADOW_WAYPOINT_SPEED_M_S * action_vy
        ),
        **prediction,
    }


def _goal_intent(telemetry: dict, target: tuple[float, ...]) -> np.ndarray:
    x = float(telemetry.get("stateEstimate.x", 0.0))
    y = float(telemetry.get("stateEstimate.y", 0.0))
    z = float(telemetry.get("stateEstimate.z", 0.0))
    yaw = radians(float(telemetry.get("stateEstimate.yaw", 0.0)))
    dx, dy, dz = target[0] - x, target[1] - y, target[2] - z
    distance = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    inverse = 1.0 / distance if distance > 1.0e-6 else 0.0
    body_x = cos(yaw) * dx + sin(yaw) * dy
    body_y = -sin(yaw) * dx + cos(yaw) * dy
    yaw_error = target[3] - yaw
    return np.asarray(
        (
            body_x * inverse,
            body_y * inverse,
            dz * inverse,
            np.clip(distance / 4.0, 0.0, 1.0),
            sin(yaw_error),
            cos(yaw_error),
        ),
        dtype=np.float32,
    )


def _summary(
    checkpoint: Path,
    rows: list[dict],
    target: tuple[float, ...],
) -> dict:
    actions = np.asarray(
        [
            [row[f"action_{axis}"] for axis in ("vx", "vy", "vz", "yaw")]
            for row in rows
        ],
        dtype=np.float32,
    )
    inference = np.asarray([row["inference_ms"] for row in rows])
    return {
        "checkpoint": str(checkpoint.resolve()),
        "controls_drone": False,
        "monitor_only": True,
        "processed_frames": len(rows),
        "all_actions_finite": bool(np.isfinite(actions).all()),
        "max_abs_action": float(np.abs(actions).max()),
        "lateral_action_p95": float(np.percentile(np.abs(actions[:, 1]), 95)),
        "lateral_residual_p95_m_s": float(
            np.percentile(
                np.abs(actions[:, 1])
                * NAVIGATION_RESIDUAL_SCALE
                * SHADOW_WAYPOINT_SPEED_M_S,
                95,
            )
        ),
        "shadow_waypoint_speed_m_s": SHADOW_WAYPOINT_SPEED_M_S,
        "inference_ms_median": float(np.median(inference)),
        "inference_ms_max": float(inference.max()),
        "target_world": list(target),
        "live_authority_approved": False,
        "next_gate": "stationary live-camera shadow with current AI Deck profile",
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
