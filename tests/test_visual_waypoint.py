from __future__ import annotations

import hashlib
import json
from queue import Queue
from time import time
from types import SimpleNamespace

import numpy as np
import pytest

from flightrl.hardware.errors import HardwareSafetyError
from flightrl.hardware.visual_waypoint import (
    StraightWaypoint,
    VisualWaypointConfig,
    bounded_waypoint_command,
    goal_intent,
    require_visual_live_readiness,
    waypoint_envelope_abort_reason,
)
from flightrl.hardware.visual_waypoint_flight import VisualFlightConfig
from flightrl.hardware.visual_waypoint_gates import (
    VISUAL_WAYPOINT_LOG_VARIABLES,
    drain_telemetry,
    shutdown_visual_flight,
    visual_control_row,
    visual_hover_abort_reason,
)
from flightrl.hardware.visual_waypoint_mission import (
    VisualMissionState,
    run_waypoint_sequence,
    warmup_visual_policy,
)


def test_straight_waypoint_intent_tracks_body_frame() -> None:
    config = VisualWaypointConfig(distance_m=0.30)
    waypoint = StraightWaypoint.from_pose(1.0, 2.0, 90.0, config)

    intent = goal_intent((1.0, 2.0, config.height_m), 90.0, waypoint)

    np.testing.assert_allclose(
        intent,
        np.asarray((1.0, 0.0, 0.0, 0.075, 0.0, 1.0), dtype=np.float32),
        atol=1.0e-6,
    )


def test_visual_policy_has_only_smoothed_lateral_authority() -> None:
    config = VisualWaypointConfig()
    waypoint = StraightWaypoint.from_pose(0.0, 0.0, 0.0, config)

    first = bounded_waypoint_command(
        (0.0, 0.0, config.height_m),
        0.0,
        waypoint,
        action_vy=1.0,
        previous_residual_m_s=0.0,
        config=config,
    )
    second = bounded_waypoint_command(
        (0.0, 0.0, config.height_m),
        0.0,
        waypoint,
        action_vy=1.0,
        previous_residual_m_s=first.policy_vy_m_s,
        config=config,
    )

    assert config.policy_authority_m_s == pytest.approx(0.0192)
    assert first.vx_body_m_s == pytest.approx(config.base_speed_m_s)
    assert first.policy_vy_m_s == pytest.approx(0.006)
    assert second.policy_vy_m_s == pytest.approx(0.012)


def test_positive_policy_lateral_matches_crazyflie_left_axis() -> None:
    config = VisualWaypointConfig()
    waypoint = StraightWaypoint.from_pose(0.0, 0.0, 0.0, config)

    command = bounded_waypoint_command(
        (0.0, 0.0, config.height_m),
        0.0,
        waypoint,
        action_vy=1.0,
        previous_residual_m_s=config.policy_authority_m_s,
        config=config,
    )

    assert command.vy_body_m_s > 0.0


def test_visual_waypoint_envelope_rejects_cross_track_drift() -> None:
    config = VisualWaypointConfig()
    waypoint = StraightWaypoint.from_pose(0.0, 0.0, 0.0, config)

    reason = waypoint_envelope_abort_reason(
        (0.10, 0.19, config.height_m),
        waypoint,
        config,
    )

    assert reason is not None
    assert reason.startswith("cross_track_gt_")


def test_readiness_requires_matching_checkpoint_and_passed_gates(tmp_path) -> None:
    checkpoint = tmp_path / "policy.bin"
    checkpoint.write_bytes(b"visual-policy")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    training = tmp_path / "training.json"
    shadow = tmp_path / "shadow.json"
    training.write_text(
        json.dumps(
            {
                "checkpoint_sha256": digest,
                "simulation_gate": {"passed": True},
            }
        )
    )
    shadow.write_text(
        json.dumps(
            {
                "checkpoint_sha256": digest,
                "next_live_shadow_gate_passed": True,
                "controls_drone": False,
            }
        )
    )

    readiness = require_visual_live_readiness(checkpoint, training, shadow)

    assert readiness["checkpoint_sha256"] == digest
    assert readiness["simulation_gate_passed"] is True
    assert readiness["stationary_shadow_gate_passed"] is True


def test_readiness_rejects_actuating_shadow_report(tmp_path) -> None:
    checkpoint = tmp_path / "policy.bin"
    checkpoint.write_bytes(b"visual-policy")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    training = tmp_path / "training.json"
    shadow = tmp_path / "shadow.json"
    training.write_text(
        json.dumps(
            {
                "checkpoint_sha256": digest,
                "simulation_gate": {"passed": True},
            }
        )
    )
    shadow.write_text(
        json.dumps(
            {
                "checkpoint_sha256": digest,
                "next_live_shadow_gate_passed": True,
                "controls_drone": True,
            }
        )
    )

    with pytest.raises(ValueError, match="must be non-actuating"):
        require_visual_live_readiness(checkpoint, training, shadow)


def test_live_config_rejects_long_unbounded_run() -> None:
    with pytest.raises(ValueError, match="max_active_s"):
        VisualFlightConfig(max_active_s=11.0)


def test_live_config_limits_waypoint_count() -> None:
    with pytest.raises(ValueError, match="waypoint_count"):
        VisualFlightConfig(waypoint_count=4)


def test_waypoint_sequence_preserves_recurrent_state(monkeypatch) -> None:
    class Motion:
        def __init__(self) -> None:
            self.stop_count = 0

        def stop(self) -> None:
            self.stop_count += 1

    class Worker:
        def __init__(self) -> None:
            self.frame_count = 0
            self.reset_count = 0
            self.intent_count = 0

        def reset(self, _intent) -> None:
            self.reset_count += 1

        def set_intent(self, _intent) -> None:
            self.intent_count += 1

        def wait_for_frames(self, minimum, *, timeout_s) -> None:
            assert timeout_s == 3.0
            self.frame_count = minimum

    latest = {
        "stateEstimate.x": 0.0,
        "stateEstimate.y": 0.0,
        "stateEstimate.z": 0.35,
        "stateEstimate.yaw": 0.0,
    }
    state = VisualMissionState()
    motion = Motion()
    worker = Worker()

    def fake_warmup(*args, **kwargs) -> None:
        worker.frame_count = 64

    def fake_active(*args, **kwargs) -> None:
        latest["stateEstimate.x"] = state.waypoint.target_x_m
        latest["stateEstimate.y"] = state.waypoint.target_y_m
        state.result = "target_reached"

    monkeypatch.setattr(
        "flightrl.hardware.visual_waypoint_mission.warmup_visual_policy",
        fake_warmup,
    )
    monkeypatch.setattr(
        "flightrl.hardware.visual_waypoint_mission.run_active_waypoint",
        fake_active,
    )

    run_waypoint_sequence(
        motion,
        worker,
        logger=None,
        latest=latest,
        state=state,
        waypoint_config=VisualWaypointConfig(
            distance_m=0.70,
            max_displacement_m=0.90,
        ),
        live_config=SimpleNamespace(
            waypoint_count=3,
            camera_timeout_s=3.0,
        ),
        rows=[],
    )

    assert state.result == "mission_complete"
    assert state.completed_waypoints == 3
    assert [waypoint.target_x_m for waypoint in state.waypoints] == pytest.approx(
        [0.70, 1.40, 2.10]
    )
    assert worker.reset_count == 1
    assert worker.intent_count == 2
    assert motion.stop_count == 3


def test_visual_waypoint_supports_bounded_longer_scale_check() -> None:
    config = VisualWaypointConfig(
        distance_m=0.70,
        base_speed_m_s=0.32,
        max_total_speed_m_s=0.34,
        max_displacement_m=0.90,
    )

    assert config.distance_m == pytest.approx(0.70)
    assert config.base_speed_m_s == pytest.approx(0.32)
    assert config.max_total_speed_m_s == pytest.approx(0.34)
    assert config.max_displacement_m == pytest.approx(0.90)


def test_visual_waypoint_total_speed_must_cover_base_speed() -> None:
    with pytest.raises(ValueError, match="must cover base_speed_m_s"):
        VisualWaypointConfig(
            base_speed_m_s=0.16,
            max_total_speed_m_s=0.12,
        )


def test_visual_waypoint_requires_displacement_margin() -> None:
    with pytest.raises(ValueError, match="must exceed distance_m"):
        VisualWaypointConfig(
            distance_m=0.70,
            max_displacement_m=0.70,
        )


def test_live_log_contract_contains_flow_and_actuator_diagnostics() -> None:
    assert {
        "motion.deltaX",
        "motion.deltaY",
        "motion.squal",
        "motion.shutter",
        "ctrltarget.vx",
        "ctrltarget.vy",
        "controller.cmd_roll",
        "controller.cmd_pitch",
        "motor.m1",
        "motor.m2",
        "motor.m3",
        "motor.m4",
    }.issubset(VISUAL_WAYPOINT_LOG_VARIABLES)


def test_firmware_hover_uses_tighter_displacement_gate() -> None:
    prediction = {
        "worker_host_time_s": 0.0,
        "frame_mean": 32.0,
        "input_contrast_std": 0.5,
        "dropped_frames": 0,
        "action_vx": 0.0,
        "action_vy": 0.0,
        "action_vz": 0.0,
        "action_yaw": 0.0,
    }
    telemetry = {
        "stateEstimate.x": 0.081,
        "stateEstimate.y": 0.0,
        "stateEstimate.z": 0.55,
        "stateEstimate.vx": 0.0,
        "stateEstimate.vy": 0.0,
    }
    live = VisualFlightConfig()
    prediction["worker_host_time_s"] = time()

    reason = visual_hover_abort_reason(
        telemetry,
        prediction,
        (0.0, 0.0),
        VisualWaypointConfig(),
        live,
    )

    assert reason == "firmware_hover_displacement_gt_0.08m:0.081"


def test_active_control_row_overrides_shadow_markers() -> None:
    row = visual_control_row(
        "active",
        {"stateEstimate.z": 0.55},
        {"controls_drone": False, "monitor_only": True},
        controls_drone=True,
    )

    assert row["controls_drone"] is True
    assert row["monitor_only"] is False


def test_shutdown_disarms_even_when_landing_fails(monkeypatch) -> None:
    calls = []

    class Motion:
        def start_linear_motion(self, *_args, **_kwargs):
            calls.append("zero")

        def stop(self):
            calls.append("hover")

        def land(self, **_kwargs):
            calls.append("land")
            raise RuntimeError("radio loss")

    class Commander:
        def send_stop_setpoint(self):
            calls.append("stop")

        def send_notify_setpoint_stop(self):
            calls.append("notify")

    monkeypatch.setattr(
        "flightrl.hardware.visual_waypoint_gates.disarm_crazyflie_after_flight",
        lambda _cf: calls.append("disarm"),
    )
    monkeypatch.setattr(
        "flightrl.hardware.visual_waypoint_gates.sleep",
        lambda _seconds: None,
    )

    with pytest.raises(HardwareSafetyError, match="landing sequence failed"):
        shutdown_visual_flight(
            object(),
            Commander(),
            Motion(),
            airborne=True,
            landing_velocity_m_s=0.15,
        )

    assert calls[-3:] == ["stop", "notify", "disarm"]


def test_warmup_waits_for_first_post_reset_frame() -> None:
    class Worker:
        frame_count = 0
        waited = False

        def wait_for_frames(self, count, *, timeout_s):
            assert count == 1
            assert timeout_s == 3.0
            self.waited = True
            self.frame_count = 1

    worker = Worker()
    warmup_visual_policy(
        worker,
        logger=None,
        latest={},
        waypoint=None,
        waypoint_config=None,
        live_config=SimpleNamespace(
            warmup_frames=1,
            warmup_timeout_s=5.0,
            camera_timeout_s=3.0,
        ),
        rows=[],
    )

    assert worker.waited is True


def test_warmup_turns_missing_post_reset_frame_into_safety_abort() -> None:
    class Worker:
        frame_count = 0

        def wait_for_frames(self, _count, *, timeout_s):
            raise TimeoutError(f"no frame in {timeout_s}")

    with pytest.raises(HardwareSafetyError, match="no post-reset warmup frame"):
        warmup_visual_policy(
            Worker(),
            logger=None,
            latest={},
            waypoint=None,
            waypoint_config=None,
            live_config=SimpleNamespace(
                warmup_frames=1,
                warmup_timeout_s=5.0,
                camera_timeout_s=3.0,
            ),
            rows=[],
        )


def test_takeoff_telemetry_drain_keeps_latest_altitude() -> None:
    class Logger:
        DISCONNECT_EVENT = "disconnect"
        _queue = Queue()

    logger = Logger()
    logger._queue.put((1, {"stateEstimate.z": 0.20}, None))
    logger._queue.put((2, {"stateEstimate.z": 0.55}, None))
    latest = {}

    drained = drain_telemetry(logger, latest)

    assert drained == 2
    assert latest["stateEstimate.z"] == pytest.approx(0.55)
    assert latest["crazyflie_time_ms"] == 2
