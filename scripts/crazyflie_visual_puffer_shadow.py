from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from flightrl.hardware.aideck_stream import AIDECK_GRAY4_FORMAT, AiDeckUdpStream
from flightrl.puffer4_vision_runtime import VisualPufferShadow
from flightrl.puffer4_vision_sections import NAVIGATION_RESIDUAL_SCALE


SHADOW_WAYPOINT_SPEED_M_S = 0.08
WARMUP_FRAMES = 64
CONTROL_PERIOD_MS = 1_000.0 / 65.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a visual Puffer checkpoint on live AI Deck frames without control"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=650)
    parser.add_argument("--target-distance-m", type=float, default=3.1)
    parser.add_argument("--aideck-host", default="192.168.4.1")
    parser.add_argument("--aideck-port", type=int, default=5000)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--bind-port", type=int, default=5001)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    args = parser.parse_args()
    if args.frames <= WARMUP_FRAMES:
        parser.error(f"--frames must be greater than {WARMUP_FRAMES}")

    shadow = VisualPufferShadow(args.checkpoint)
    intent = np.asarray(
        (1.0, 0.0, 0.0, min(args.target_distance_m / 4.0, 1.0), 0.0, 1.0),
        dtype=np.float32,
    )
    stream = AiDeckUdpStream(
        host=args.aideck_host,
        port=args.aideck_port,
        bind_host=args.bind_host,
        bind_port=args.bind_port,
        timeout_s=args.timeout_s,
    )
    rows = []
    started = perf_counter()
    with stream:
        for frame in stream.frames(args.frames):
            prediction = shadow.step(frame.pixels, intent)
            rows.append(
                {
                    "frame_index": frame.index,
                    "host_time_s": frame.host_time_s,
                    "width": frame.width,
                    "height": frame.height,
                    "format": frame.format,
                    "frame_mean": float(frame.pixels.mean()),
                    **prediction,
                }
            )
    elapsed = perf_counter() - started
    _write_rows(args.output, rows)
    summary = _summary(args, rows, stream.dropped_frames, elapsed)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"output={args.output}")
    print(f"summary={summary_path}")


def _summary(
    args: argparse.Namespace,
    rows: list[dict],
    dropped_frames: int,
    elapsed_s: float,
) -> dict:
    actions = np.asarray(
        [
            [row[f"action_{axis}"] for axis in ("vx", "vy", "vz", "yaw")]
            for row in rows
        ],
        dtype=np.float32,
    )
    inference = np.asarray([row["inference_ms"] for row in rows])
    contrast = np.asarray([row["input_contrast_std"] for row in rows])
    delta = np.asarray([row["input_delta_mae"] for row in rows])
    motion = np.asarray([row["input_motion_fraction"] for row in rows])
    gated_actions = actions[WARMUP_FRAMES:]
    lateral = np.abs(gated_actions[:, 1])
    bounded_lateral = lateral * NAVIGATION_RESIDUAL_SCALE * SHADOW_WAYPOINT_SPEED_M_S
    off_axis_p95 = float(
        np.percentile(np.abs(gated_actions[:, (0, 2, 3)]), 95)
    )
    stream_fps = len(rows) / elapsed_s
    behavior_passed = bool(
        np.isfinite(gated_actions).all()
        and float(bounded_lateral.max()) <= 0.05
        and off_axis_p95 <= 0.05
    )
    low_light_passed = bool(
        float(np.mean([row["frame_mean"] for row in rows])) >= 8.0
        and float(np.median(contrast)) >= 0.10
        and float(np.percentile(delta[1:], 95)) <= 0.05
        and float(np.percentile(motion[1:], 95)) <= 0.10
    )
    profile_passed = bool(
        rows[0]["width"] == 64
        and rows[0]["height"] == 48
        and rows[0]["format"] == AIDECK_GRAY4_FORMAT
        and stream_fps >= 50.0
        and dropped_frames == 0
    )
    inference_passed = float(inference.max()) <= CONTROL_PERIOD_MS
    return {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "controls_drone": False,
        "monitor_only": True,
        "frames": len(rows),
        "warmup_frames": WARMUP_FRAMES,
        "post_warmup_frames": len(gated_actions),
        "dropped_frames": dropped_frames,
        "stream_fps": stream_fps,
        "frame_width": rows[0]["width"],
        "frame_height": rows[0]["height"],
        "frame_mean": float(np.mean([row["frame_mean"] for row in rows])),
        "all_actions_finite": bool(np.isfinite(actions).all()),
        "max_abs_action": float(np.abs(actions).max()),
        "lateral_action_p95": float(np.percentile(lateral, 95)),
        "bounded_lateral_p95_m_s": float(np.percentile(bounded_lateral, 95)),
        "bounded_lateral_max_m_s": float(bounded_lateral.max()),
        "off_axis_action_p95": off_axis_p95,
        "inference_ms_median": float(np.median(inference)),
        "inference_ms_max": float(inference.max()),
        "inference_deadline_gate_passed": inference_passed,
        "input_contrast_std_median": float(np.median(contrast)),
        "input_delta_mae_p95": float(np.percentile(delta[1:], 95)),
        "input_motion_fraction_p95": float(np.percentile(motion[1:], 95)),
        "low_light_input_gate_passed": low_light_passed,
        "shadow_behavior_gate_passed": behavior_passed,
        "camera_profile_gate_passed": profile_passed,
        "next_live_shadow_gate_passed": (
            behavior_passed
            and low_light_passed
            and profile_passed
            and inference_passed
        ),
        "live_authority_approved": False,
        "next_gate": (
            "restore and verify the frame-safe 64x48 gray4 profile"
            if behavior_passed and low_light_passed and not profile_passed
            else "review shadow actions before any bounded flight authority"
        ),
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
