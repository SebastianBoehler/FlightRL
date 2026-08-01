from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from time import time

import numpy as np

from flightrl.hardware.aideck_stream import AiDeckUdpStream
from flightrl.hardware.avoidance_live import next_log_sample
from flightrl.hardware.cflib_bridge import (
    require_cflib,
    sync_crazyflie_context,
)
from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.telemetry import (
    build_log_configs,
    with_available_log_variables,
)
from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
)
from flightrl.puffer4_door_runtime import DoorPufferShadow
from flightrl.puffer4_door_self_mask import DoorSelfMaskedGrounder
from flightrl.puffer4_door_shadow_detector import (
    build_approved_shadow_grounder,
)
from flightrl.puffer4_door_shadow_io import (
    REQUIRED_TELEMETRY,
    configure_shadow_logging,
    require_telemetry_contract,
    telemetry_csv_fields,
)
from flightrl.semantic import AsyncGroundingPipeline


def collect_live_rows(
    shadow: DoorPufferShadow,
    args,
    *,
    hardware_config_path: str | Path | None = None,
) -> tuple[list[dict], int]:
    grounder = build_approved_shadow_grounder(args.device)
    stream = AiDeckUdpStream(
        host=args.aideck_host,
        port=args.aideck_port,
        bind_host=args.bind_host,
        bind_port=args.bind_port,
        timeout_s=args.camera_timeout_s,
    )
    pipeline = AsyncGroundingPipeline(
        stream,
        DoorSelfMaskedGrounder(grounder),
        args.prompt,
    )
    hardware = configure_shadow_logging(
        load_hardware_config(hardware_config_path or args.hardware_config)
    )
    modules = require_cflib()
    rows: list[dict] = []
    latest: dict[str, float] = {}
    last_frame = 0
    try:
        pipeline.start()
        pipeline.wait_for_frame(args.camera_timeout_s)
        pipeline.wait_for_result(args.camera_timeout_s)
        with sync_crazyflie_context(hardware, modules) as scf:
            config = with_available_log_variables(scf, hardware)
            require_telemetry_contract(config.logging.variables)
            log_configs = build_log_configs(modules, config)
            deadline = time() + args.duration_s
            with modules.sync_logger_cls(scf, log_configs) as logger:
                while time() < deadline:
                    sample = _next_sample(logger, args.log_timeout_s)
                    if sample is None:
                        continue
                    _, values, _ = sample
                    latest.update(
                        {key: float(value) for key, value in values.items()}
                    )
                    frame = pipeline.latest_frame()
                    if frame is None or frame.index == last_frame:
                        continue
                    if not all(key in latest for key in REQUIRED_TELEMETRY):
                        continue
                    grounding = pipeline.latest()
                    detection, age_s, detector_ms = latest_detection(
                        grounding,
                        now_s=frame.host_time_s,
                        policy_frame_index=frame.index,
                    )
                    assert grounding is not None
                    grounding_frame, _ = grounding
                    prediction = shadow.step(
                        frame.pixels,
                        latest,
                        detection=detection,
                        detection_age_s=age_s,
                        executed_previous_action=np.zeros(
                            2,
                            dtype=np.float32,
                        ),
                    )
                    rows.append(
                        _capture_row(
                            frame,
                            latest,
                            detection,
                            age_s,
                            detector_ms,
                            grounding_frame.index,
                            prediction,
                        )
                    )
                    last_frame = frame.index
    finally:
        pipeline.close()
    if not rows:
        raise RuntimeError("live shadow produced no synchronized frame/telemetry rows")
    dropped_frames = stream.dropped_frames
    for row in rows:
        row["stream_dropped_frames"] = dropped_frames
    return rows, dropped_frames


def latest_detection(
    latest,
    *,
    now_s: float,
    policy_frame_index: int | None = None,
):
    if latest is None:
        return None, None, None
    source_frame, result = latest
    source_index = int(
        source_frame.index
        if hasattr(source_frame, "index")
        else source_frame
    )
    result_index = int(getattr(result, "frame_index", source_index))
    if (
        result_index != source_index
        or (
            policy_frame_index is not None
            and source_index > policy_frame_index
        )
        or float(result.frame_host_time_s) > now_s
    ):
        raise RuntimeError(
            "grounding result is newer than policy frame"
        )
    age_s = now_s - result.frame_host_time_s
    maximum_age_s = (
        FIXED_DOOR_EVIDENCE_AGE_CONTRACT.maximum_evidence_age_s
    )
    detection = result.best if age_s < maximum_age_s else None
    return detection, age_s, result.inference_ms


def dry_run_row(shadow: DoorPufferShadow) -> dict:
    telemetry = {
        "stateEstimate.x": 0.0,
        "stateEstimate.y": 0.0,
        "stateEstimate.z": 0.8,
        "stateEstimate.vx": 0.0,
        "stateEstimate.vy": 0.0,
        "stateEstimate.vz": 0.0,
        "stateEstimate.roll": 0.0,
        "stateEstimate.pitch": 0.0,
        "stateEstimate.yaw": 0.0,
        "gyro.x": 0.0,
        "gyro.y": 0.0,
        "gyro.z": 0.0,
    }
    prediction = shadow.step(
        np.full((96, 128), 51, dtype=np.uint8),
        telemetry,
        detection=None,
        executed_previous_action=np.zeros(2, dtype=np.float32),
    )
    return {
        "frame_index": 0,
        "frame_host_time_s": 0.0,
        "frame_width": 128,
        "frame_height": 96,
        "grounding_age_s": None,
        "grounding_inference_ms": 0.0,
        "grounding_result_frame_index": 0,
        "detection": None,
        "stream_dropped_frames": 0,
        **prediction,
    }


def _next_sample(logger, timeout_s: float):
    try:
        return next_log_sample(logger, timeout_s=timeout_s)
    except StopIteration as exc:
        raise RuntimeError(
            "Crazyflie telemetry disconnected during shadow capture"
        ) from exc


def _capture_row(
    frame,
    telemetry,
    detection,
    age_s,
    detector_ms,
    grounding_frame_index,
    prediction,
) -> dict:
    return {
        "frame_index": frame.index,
        "frame_host_time_s": frame.host_time_s,
        "frame_width": frame.width,
        "frame_height": frame.height,
        "frame_mean": float(frame.pixels.mean()),
        "grounding_age_s": age_s,
        "grounding_inference_ms": detector_ms,
        "grounding_result_frame_index": grounding_frame_index,
        "detection": (
            None if detection is None else json.dumps(asdict(detection))
        ),
        "battery_v": telemetry.get("pm.vbat"),
        **telemetry_csv_fields(telemetry),
        **prediction,
    }
