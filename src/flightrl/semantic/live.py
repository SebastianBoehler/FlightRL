from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from time import sleep, time
from typing import Any

from flightrl.hardware.aideck_stream import AiDeckFrame
from flightrl.hardware.avoidance_live import safety_abort_reason
from flightrl.hardware.cflib_bridge import require_cflib, sync_crazyflie_context
from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.errors import HardwareSafetyError
from flightrl.hardware.motion import (
    arm_crazyflie_for_flight,
    disarm_crazyflie_after_flight,
    install_legacy_hover_warning_filter,
    reset_crazyflie_estimator,
)
from flightrl.hardware.preflight import (
    inspect_decks,
    require_supervisor_allows_flight,
)
from flightrl.hardware.telemetry import (
    TelemetryCsvWriter,
    TelemetrySample,
    build_log_configs,
    with_available_log_variables,
)

from .controller import (
    DiscoveryConfig,
    DiscoveryController,
    DiscoveryPhase,
)
from .dataset import SemanticRunWriter
from .worker import AsyncGroundingPipeline


SEMANTIC_LOG_VARIABLES = (
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stateEstimate.roll",
    "stateEstimate.pitch",
    "stateEstimate.yaw",
    "gyro.x",
    "gyro.y",
    "gyro.z",
    "pm.vbat",
    "pm.batteryLevel",
    "sys.isFlying",
    "sys.isTumbled",
    "motor.m1",
    "motor.m2",
    "motor.m3",
    "motor.m4",
)


@dataclass(frozen=True, slots=True)
class SemanticFlightConfig:
    height_m: float = 0.3
    max_duration_s: float = 45.0
    min_frame_width: int = 128
    min_frame_mean: float = 8.0
    allow_reposition: bool = False

    def __post_init__(self) -> None:
        if not 0.1 <= self.height_m <= 0.8:
            raise ValueError("semantic flight height must be in [0.1, 0.8] meters")
        if not 1.0 <= self.max_duration_s <= 60.0:
            raise ValueError("semantic flight duration must be in [1, 60] seconds")
        if self.min_frame_width <= 0:
            raise ValueError("minimum semantic frame width must be positive")
        if not 0.0 <= self.min_frame_mean <= 255.0:
            raise ValueError("minimum semantic frame mean must be in [0, 255]")


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
            writer.write(frame, result, controls_drone=False)
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


def run_semantic_flight(
    pipeline: AsyncGroundingPipeline,
    writer: SemanticRunWriter,
    *,
    hardware_config_path: str | Path,
    flight: SemanticFlightConfig,
    discovery: DiscoveryConfig,
) -> dict[str, Any]:
    config = load_hardware_config(hardware_config_path)
    config = replace(
        config,
        logging=replace(config.logging, variables=SEMANTIC_LOG_VARIABLES),
    )
    modules = require_cflib()
    install_legacy_hover_warning_filter()
    latest_telemetry: dict[str, float] = {}
    summary: dict[str, Any] = {
        "mode": "flight",
        "processed_frames": 0,
        "frames_with_detection": 0,
        "telemetry_samples": 0,
        "abort_reason": None,
        "final_phase": None,
    }
    telemetry_path = writer.output_dir / "telemetry.csv"

    with sync_crazyflie_context(config, modules) as scf:
        _require_flow_deck(scf, config)
        require_supervisor_allows_flight(scf, modules, config)
        log_config = with_available_log_variables(scf, config)
        log_blocks = build_log_configs(modules, log_config)
        motion = modules.motion_commander_cls(scf, default_height=flight.height_m)
        commander = scf.cf.commander
        airborne = False
        origin_xy: tuple[float, float] | None = None
        controller: DiscoveryController | None = None
        last_frame_index = -1
        try:
            reset_crazyflie_estimator(scf.cf)
            arm_crazyflie_for_flight(scf.cf)
            motion.take_off(
                height=flight.height_m,
                velocity=config.safety.velocity_m_s,
            )
            airborne = True
            with TelemetryCsvWriter(
                telemetry_path,
                variables=log_config.logging.variables,
            ) as telemetry_writer:
                with modules.sync_logger_cls(scf, log_blocks) as logger:
                    for crazyflie_time_ms, values, _conf in logger:
                        latest_telemetry.update(
                            {key: float(value) for key, value in values.items()}
                        )
                        if "stateEstimate.yaw" not in values:
                            continue
                        now_s = time()
                        telemetry_writer.write_sample(
                            TelemetrySample(
                                now_s,
                                int(crazyflie_time_ms),
                                latest_telemetry.copy(),
                            )
                        )
                        summary["telemetry_samples"] += 1
                        abort = safety_abort_reason(
                            latest_telemetry,
                            target_height_m=flight.height_m,
                        )
                        if abort is not None:
                            summary["abort_reason"] = abort
                            break
                        position = (
                            latest_telemetry.get("stateEstimate.x", 0.0),
                            latest_telemetry.get("stateEstimate.y", 0.0),
                        )
                        if origin_xy is None:
                            origin_xy = position
                            controller = DiscoveryController(
                                discovery,
                                start_time_s=now_s,
                            )
                        assert controller is not None
                        grounded = pipeline.latest()
                        result = grounded[1] if grounded is not None else None
                        command = controller.step(
                            now_s=now_s,
                            grounding=result,
                            position_xy_m=position,
                            origin_xy_m=origin_xy,
                            yaw_deg=latest_telemetry.get("stateEstimate.yaw", 0.0),
                        )
                        motion.start_linear_motion(
                            command.vx_body_m_s,
                            command.vy_body_m_s,
                            0.0,
                            rate_yaw=command.yawrate_deg_s,
                        )
                        if grounded is not None and grounded[0].index != last_frame_index:
                            writer.write(
                                grounded[0],
                                grounded[1],
                                command=command,
                                telemetry=latest_telemetry,
                                controls_drone=True,
                            )
                            last_frame_index = grounded[0].index
                            summary["processed_frames"] += 1
                            summary["frames_with_detection"] += int(
                                grounded[1].best is not None
                            )
                        summary["final_phase"] = command.phase.value
                        if command.phase in {
                            DiscoveryPhase.COMPLETE,
                            DiscoveryPhase.TIMEOUT,
                        }:
                            break
        finally:
            if airborne:
                motion.stop()
                sleep(0.5)
                motion.land(velocity=config.safety.velocity_m_s)
            commander.send_stop_setpoint()
            commander.send_notify_setpoint_stop()
            disarm_crazyflie_after_flight(scf.cf)
    processed = summary["processed_frames"]
    summary["detection_rate"] = (
        summary["frames_with_detection"] / processed if processed else 0.0
    )
    return summary


def write_summary(output_dir: str | Path, summary: dict[str, Any]) -> Path:
    path = Path(output_dir) / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def _require_flow_deck(scf, config) -> None:
    report = inspect_decks(scf, config)
    if not report.ok:
        details = "; ".join((*report.warnings, *(f"{k}={v}" for k, v in report.details.items())))
        raise HardwareSafetyError(f"Flow Deck preflight failed: {details}")
