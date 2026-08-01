from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import hypot
from time import time

from .errors import HardwareSafetyError
from .visual_waypoint import (
    StraightWaypoint,
    bounded_waypoint_command,
)
from .visual_waypoint_gates import (
    intent_from_telemetry,
    position_from_telemetry,
    update_telemetry,
    visual_control_row,
    visual_live_abort_reason,
)


@dataclass(slots=True)
class VisualMissionState:
    result: str = "preflight"
    abort_reason: str | None = None
    waypoint: StraightWaypoint | None = None
    waypoints: list[StraightWaypoint] = field(default_factory=list)
    completed_waypoints: int = 0
    max_policy_vy_m_s: float = 0.0
    flight_commanded: bool = False
    hover_max_displacement_m: float = 0.0
    hover_max_speed_m_s: float = 0.0


def run_waypoint_sequence(
    motion,
    worker,
    logger,
    latest,
    state,
    waypoint_config,
    live_config,
    rows,
) -> None:
    mission_origin = (
        float(latest["stateEstimate.x"]),
        float(latest["stateEstimate.y"]),
    )
    mission_limit_m = (
        waypoint_config.distance_m * live_config.waypoint_count
        + waypoint_config.max_displacement_m
        - waypoint_config.distance_m
    )
    for waypoint_index in range(1, live_config.waypoint_count + 1):
        if waypoint_index > 1:
            motion.stop()
        state.waypoint = StraightWaypoint.from_pose(
            latest["stateEstimate.x"],
            latest["stateEstimate.y"],
            latest["stateEstimate.yaw"],
            waypoint_config,
        )
        state.waypoints.append(state.waypoint)
        intent = intent_from_telemetry(latest, state.waypoint)
        if waypoint_index == 1:
            worker.reset(intent)
            state.result = "warmup"
            warmup_visual_policy(
                worker,
                logger,
                latest,
                state.waypoint,
                waypoint_config,
                live_config,
                rows,
                waypoint_index,
            )
        else:
            frame_count = worker.frame_count
            worker.set_intent(intent)
            worker.wait_for_frames(
                frame_count + 1,
                timeout_s=live_config.camera_timeout_s,
            )
        state.result = "active"
        run_active_waypoint(
            motion,
            worker,
            logger,
            latest,
            state,
            waypoint_config,
            live_config,
            rows,
            waypoint_index=waypoint_index,
            mission_origin=mission_origin,
            mission_limit_m=mission_limit_m,
        )
        if state.result != "target_reached":
            return
        state.completed_waypoints += 1
    motion.stop()
    if live_config.waypoint_count > 1:
        state.result = "mission_complete"


def warmup_visual_policy(
    worker,
    logger,
    latest,
    waypoint,
    waypoint_config,
    live_config,
    rows,
    waypoint_index=1,
) -> None:
    deadline = time() + live_config.warmup_timeout_s
    try:
        worker.wait_for_frames(1, timeout_s=live_config.camera_timeout_s)
    except TimeoutError as exc:
        raise HardwareSafetyError(
            "visual policy produced no post-reset warmup frame"
        ) from exc
    while worker.frame_count < live_config.warmup_frames:
        if time() >= deadline:
            raise HardwareSafetyError("visual recurrent warmup timed out")
        update_telemetry(logger, latest, live_config.log_timeout_s)
        prediction = worker.snapshot()
        abort = visual_live_abort_reason(
            latest,
            prediction,
            waypoint,
            waypoint_config,
            live_config,
        )
        if abort is not None:
            raise HardwareSafetyError(f"warmup safety abort: {abort}")
        worker.set_intent(intent_from_telemetry(latest, waypoint))
        row = visual_control_row(
            "warmup",
            latest,
            prediction,
            controls_drone=False,
        )
        row["waypoint_index"] = waypoint_index
        rows.append(row)


def run_active_waypoint(
    motion,
    worker,
    logger,
    latest,
    state,
    waypoint_config,
    live_config,
    rows,
    *,
    waypoint_index,
    mission_origin,
    mission_limit_m,
) -> None:
    assert state.waypoint is not None
    previous_residual = 0.0
    active_started = time()
    while time() - active_started < live_config.max_active_s:
        update_telemetry(logger, latest, live_config.log_timeout_s)
        prediction = worker.snapshot()
        state.abort_reason = visual_live_abort_reason(
            latest,
            prediction,
            state.waypoint,
            waypoint_config,
            live_config,
        )
        if state.abort_reason is None:
            state.abort_reason = _mission_abort_reason(
                latest,
                mission_origin,
                mission_limit_m,
            )
        if state.abort_reason is not None:
            state.result = "safety_abort"
            return
        worker.set_intent(intent_from_telemetry(latest, state.waypoint))
        command = bounded_waypoint_command(
            position_from_telemetry(latest),
            latest["stateEstimate.yaw"],
            state.waypoint,
            float(prediction["action_vy"]),
            previous_residual,
            waypoint_config,
        )
        previous_residual = command.policy_vy_m_s
        state.max_policy_vy_m_s = max(
            state.max_policy_vy_m_s,
            abs(previous_residual),
        )
        row = visual_control_row(
            "active",
            latest,
            prediction,
            command={
                "waypoint_index": waypoint_index,
                **asdict(command),
            },
            controls_drone=True,
        )
        rows.append(row)
        if command.target_distance_m <= waypoint_config.target_radius_m:
            state.result = "target_reached"
            return
        motion.start_linear_motion(
            command.vx_body_m_s,
            command.vy_body_m_s,
            0.0,
            rate_yaw=0.0,
        )
    state.result = "active_timeout"
    state.abort_reason = f"waypoint_{waypoint_index}_active_timeout"


def _mission_abort_reason(latest, origin_xy_m, limit_m) -> str | None:
    displacement = hypot(
        float(latest["stateEstimate.x"]) - origin_xy_m[0],
        float(latest["stateEstimate.y"]) - origin_xy_m[1],
    )
    if displacement > limit_m:
        return f"mission_displacement_gt_{limit_m:.2f}m:{displacement:.3f}"
    return None
