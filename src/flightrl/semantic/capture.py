from __future__ import annotations

from dataclasses import asdict
from statistics import median
from time import sleep, time
from typing import Any

from flightrl.hardware.aideck_stream import AiDeckFrame

from .dataset import SemanticRunWriter
from .worker import AsyncGroundingPipeline


def require_semantic_frame(
    frame: AiDeckFrame,
    *,
    min_width: int,
    min_mean: float,
) -> None:
    mean = float(frame.pixels.mean())
    if frame.width < min_width:
        raise RuntimeError(
            f"semantic camera requires width >= {min_width}, got "
            f"{frame.width}x{frame.height}; flash the semantic JPEG profile"
        )
    if mean < min_mean:
        raise RuntimeError(
            f"semantic camera frame is too dark: mean={mean:.2f} < {min_mean:.2f}"
        )


def collect_camera_only(
    pipeline: AsyncGroundingPipeline,
    writer: SemanticRunWriter,
    *,
    duration_s: float,
    policy_shadow=None,
) -> dict[str, Any]:
    deadline = time() + duration_s
    last_frame_index = -1
    written = 0
    detected = 0
    inference_ms: list[float] = []
    while time() < deadline:
        latest = pipeline.latest()
        if latest is not None and latest[0].index != last_frame_index:
            frame, result = latest
            writer.write(
                frame,
                result,
                policy_shadow=_policy_shadow(policy_shadow, frame, result),
                controls_drone=False,
            )
            last_frame_index = frame.index
            written += 1
            detected += int(result.best is not None)
            inference_ms.append(result.inference_ms)
        sleep(0.02)
    return {
        "mode": "camera",
        "processed_frames": written,
        "frames_with_detection": detected,
        "detection_rate": detected / written if written else 0.0,
        "inference_ms_median": median(inference_ms) if inference_ms else None,
        "inference_ms_max": max(inference_ms, default=None),
    }


def _policy_shadow(policy_shadow, frame, result) -> dict:
    if policy_shadow is None:
        return {}
    detection = None if result.best is None else asdict(result.best)
    return policy_shadow.step(
        frame=frame.pixels,
        telemetry={},
        prompt=result.prompt,
        detection=detection,
    )
