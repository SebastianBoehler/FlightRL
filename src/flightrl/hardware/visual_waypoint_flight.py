from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from time import time
from typing import Mapping

import numpy as np

from .aideck_stream import AiDeckUdpStream
from .cflib_bridge import require_cflib, sync_crazyflie_context
from .config import load_hardware_config
from .errors import HardwareSafetyError
from .motion import (
    arm_crazyflie_for_flight,
    install_legacy_hover_warning_filter,
)
from .preflight import require_supervisor_allows_flight
from .telemetry import (
    build_log_configs,
    with_available_log_variables,
)
from .visual_policy_worker import VisualPolicyWorker, VisualPolicyWorkerError
from .visual_waypoint import VisualWaypointConfig
from .visual_waypoint_gates import (
    VISUAL_WAYPOINT_LOG_VARIABLES,
    require_camera_gate,
    require_flow_deck,
    shutdown_visual_flight,
    update_telemetry,
    visual_control_row,
    visual_hover_abort_reason,
    wait_for_takeoff,
    wait_for_telemetry,
)
from .visual_waypoint_mission import (
    VisualMissionState,
    run_waypoint_sequence,
)


@dataclass(frozen=True, slots=True)
class VisualFlightConfig:
    baseline_hover_only: bool = False
    waypoint_count: int = 1
    settle_hover_s: float = 2.0
    max_hover_displacement_m: float = 0.08
    max_hover_speed_m_s: float = 0.12
    max_active_s: float = 8.0
    warmup_frames: int = 64
    warmup_timeout_s: float = 5.0
    camera_timeout_s: float = 3.0
    max_camera_age_s: float = 0.25
    max_dropped_frames: int = 5
    min_frame_mean: float = 8.0
    min_input_contrast: float = 0.10
    min_battery_v: float = 3.55
    log_timeout_s: float = 0.5

    def __post_init__(self) -> None:
        if not 1 <= self.waypoint_count <= 3:
            raise ValueError("waypoint_count must be in [1, 3]")
        if not 0.5 <= self.settle_hover_s <= 5.0:
            raise ValueError("settle_hover_s must be in [0.5, 5]")
        if not 0.03 <= self.max_hover_displacement_m <= 0.15:
            raise ValueError("max_hover_displacement_m must be in [0.03, 0.15]")
        if not 0.05 <= self.max_hover_speed_m_s <= 0.30:
            raise ValueError("max_hover_speed_m_s must be in [0.05, 0.30]")
        if not 1.0 <= self.max_active_s <= 10.0:
            raise ValueError("max_active_s must be in [1, 10]")
        if not 16 <= self.warmup_frames <= 256:
            raise ValueError("warmup_frames must be in [16, 256]")
        if not 0.05 <= self.max_camera_age_s <= 0.50:
            raise ValueError("max_camera_age_s must be in [0.05, 0.50]")
        if not 0 <= self.max_dropped_frames <= 20:
            raise ValueError("max_dropped_frames must be in [0, 20]")


def run_visual_waypoint_flight(
    checkpoint,
    hardware_config_path,
    waypoint_config: VisualWaypointConfig,
    live_config: VisualFlightConfig,
    readiness: Mapping[str, object],
) -> tuple[dict, list[dict]]:
    hardware = load_hardware_config(hardware_config_path)
    hardware = replace(
        hardware,
        logging=replace(
            hardware.logging,
            variables=VISUAL_WAYPOINT_LOG_VARIABLES,
        ),
    )
    modules = require_cflib()
    install_legacy_hover_warning_filter()
    stream = AiDeckUdpStream(timeout_s=live_config.camera_timeout_s)
    worker = VisualPolicyWorker(
        checkpoint,
        stream=stream,
        initial_intent=np.asarray(
            (1.0, 0.0, 0.0, waypoint_config.distance_m / 4.0, 0.0, 1.0), dtype=np.float32
        ),
    )
    rows: list[dict] = []
    state = VisualMissionState()
    latest: dict[str, float] = {}
    worker.start()
    try:
        camera = worker.wait_for_frames(10, timeout_s=live_config.camera_timeout_s)
        require_camera_gate(camera, live_config)
        with sync_crazyflie_context(hardware, modules) as scf:
            try:
                _run_connected(
                    scf,
                    modules,
                    hardware,
                    worker,
                    waypoint_config,
                    live_config,
                    latest,
                    rows,
                    state,
                )
            except (HardwareSafetyError, VisualPolicyWorkerError) as exc:
                state.result = "safety_abort"
                state.abort_reason = str(exc)
    finally:
        worker.stop()
    snapshot = rows[-1] if rows else {}
    return {
        **readiness,
        "result": state.result,
        "abort_reason": state.abort_reason,
        "controls_drone": any(row["controls_drone"] for row in rows),
        "flight_commanded": state.flight_commanded,
        "firmware_controls": ["takeoff", "attitude", "altitude", "landing"],
        "policy_axis_authority": (
            "none" if live_config.baseline_hover_only else "lateral_velocity_only"
        ),
        "baseline_hover_max_displacement_m": state.hover_max_displacement_m,
        "baseline_hover_max_speed_m_s": state.hover_max_speed_m_s,
        "max_policy_vy_m_s": state.max_policy_vy_m_s,
        "policy_controls_drone": state.max_policy_vy_m_s > 0.0,
        "rows": len(rows),
        "camera_frames": worker.frame_count,
        "dropped_frames": int(snapshot.get("dropped_frames", 0)),
        "waypoint": None if state.waypoint is None else asdict(state.waypoint),
        "waypoints": [asdict(waypoint) for waypoint in state.waypoints],
        "completed_waypoints": state.completed_waypoints,
        "requested_waypoints": live_config.waypoint_count,
        "config": asdict(waypoint_config),
    }, rows


def _run_connected(
    scf,
    modules,
    hardware,
    worker,
    waypoint_config,
    live_config,
    latest,
    rows,
    state,
) -> None:
    require_flow_deck(scf, hardware)
    require_supervisor_allows_flight(scf, modules, hardware)
    log_config = with_available_log_variables(scf, hardware)
    commander = scf.cf.commander
    motion = modules.motion_commander_cls(
        scf,
        default_height=waypoint_config.height_m,
    )
    airborne = False
    try:
        with modules.sync_logger_cls(
            scf,
            build_log_configs(modules, log_config),
        ) as logger:
            wait_for_telemetry(logger, latest, live_config.log_timeout_s)
            if latest.get("pm.vbat", 0.0) < live_config.min_battery_v:
                raise HardwareSafetyError(
                    f"battery below {live_config.min_battery_v:.2f} V"
                )
            arm_crazyflie_for_flight(scf.cf)
            airborne = True
            state.flight_commanded = True
            motion.take_off(
                height=waypoint_config.height_m,
                velocity=hardware.safety.velocity_m_s,
            )
            motion.stop()
            wait_for_takeoff(
                logger,
                latest,
                waypoint_config.height_m,
                live_config.log_timeout_s,
            )
            state.result = "firmware_hover"
            _settle_firmware_hover(
                motion,
                worker,
                logger,
                latest,
                waypoint_config,
                live_config,
                rows,
                state,
            )
            if live_config.baseline_hover_only:
                state.result = "baseline_hover_complete"
                return
            run_waypoint_sequence(
                motion,
                worker,
                logger,
                latest,
                state,
                waypoint_config,
                live_config,
                rows,
            )
    finally:
        shutdown_visual_flight(
            scf.cf,
            commander,
            motion,
            airborne=airborne,
            landing_velocity_m_s=hardware.safety.velocity_m_s,
        )


def _settle_firmware_hover(
    motion,
    worker,
    logger,
    latest,
    waypoint_config,
    live_config,
    rows,
    state,
) -> None:
    origin = (
        float(latest["stateEstimate.x"]),
        float(latest["stateEstimate.y"]),
    )
    motion.stop()
    deadline = time() + live_config.settle_hover_s
    while time() < deadline:
        update_telemetry(logger, latest, live_config.log_timeout_s)
        prediction = worker.snapshot()
        abort = visual_hover_abort_reason(
            latest,
            prediction,
            origin,
            waypoint_config,
            live_config,
        )
        if abort is not None:
            raise HardwareSafetyError(f"firmware hover safety abort: {abort}")
        displacement = float(
            np.hypot(
                latest["stateEstimate.x"] - origin[0],
                latest["stateEstimate.y"] - origin[1],
            )
        )
        speed = float(
            np.hypot(
                latest.get("stateEstimate.vx", 0.0),
                latest.get("stateEstimate.vy", 0.0),
            )
        )
        state.hover_max_displacement_m = max(
            state.hover_max_displacement_m,
            displacement,
        )
        state.hover_max_speed_m_s = max(state.hover_max_speed_m_s, speed)
        rows.append(
            visual_control_row(
                "firmware_hover",
                latest,
                prediction,
                controls_drone=False,
            )
        )
