from __future__ import annotations

from queue import Queue
from types import SimpleNamespace

from flightrl.hardware.avoidance_live import maybe_emergency_command, next_log_sample, safety_abort_reason, update_range_rate
from flightrl.hardware.avoidance_policy import AvoidanceCommand, RangerReading
from flightrl.hardware.ttc_policy import TTCAvoidancePolicy


def test_update_range_rate_smooths_closing_speed() -> None:
    previous = RangerReading(1.0, 2.0, 2.0, 2.0, 2.0, 0.5)
    current = RangerReading(0.8, 2.0, 2.0, 2.0, 2.0, 0.5)

    initial = update_range_rate(current, previous, 1.0, 1.1, None, alpha=0.5)
    smoothed = update_range_rate(current, previous, 1.0, 1.1, initial, alpha=0.5)

    assert initial is not None
    assert smoothed is not None
    assert initial.front_m < -1.9
    assert smoothed.front_m == initial.front_m


def test_update_range_rate_ignores_no_return_jumps() -> None:
    previous = RangerReading(4.0, 2.0, 2.0, 2.0, 2.0, 0.5)
    current = RangerReading(0.4, 2.0, 2.0, 2.0, 2.0, 0.5)

    rate = update_range_rate(current, previous, 1.0, 1.05, None, alpha=1.0, max_abs_rate_m_s=5.0)

    assert rate is not None
    assert rate.front_m == 0.0


def test_update_range_rate_clamps_implausible_speed() -> None:
    previous = RangerReading(1.0, 2.0, 2.0, 2.0, 2.0, 0.5)
    current = RangerReading(0.4, 2.0, 2.0, 2.0, 2.0, 0.5)

    rate = update_range_rate(current, previous, 1.0, 1.05, None, alpha=1.0, max_abs_rate_m_s=5.0)

    assert rate is not None
    assert rate.front_m == -5.0


def test_ttc_can_trigger_emergency_before_distance_threshold() -> None:
    args = SimpleNamespace(
        clearance_m=0.45,
        hard_clearance_m=0.08,
        height_m=0.5,
        emergency_clearance_m=0.10,
        emergency_ttc_s=0.5,
        emergency_max_speed_m_s=0.8,
        ttc_horizon_s=0.8,
        ttc_hard_s=0.15,
        ttc_gain=1.2,
        lock_height=True,
    )
    command = AvoidanceCommand(0.0, 0.0, 0.0, 0.5)
    reading = RangerReading(0.6, 2.0, 2.0, 2.0, 2.0, 0.5)
    rate = RangerReading(-1.5, 0.0, 0.0, 0.0, 0.0, 0.0)

    emergency_command, emergency = maybe_emergency_command(command, reading, rate, args)

    assert emergency is True
    assert emergency_command.vx_m_s < -0.1


def test_next_log_sample_times_out_without_blocking() -> None:
    logger = SimpleNamespace(_queue=Queue(), DISCONNECT_EVENT="DISCONNECT")

    assert next_log_sample(logger, timeout_s=0.001) is None


def test_safety_abort_detects_tumble_and_uncontrolled_attitude() -> None:
    assert safety_abort_reason({"sys.isTumbled": 1.0}, target_height_m=0.5) == "tumbled"
    assert safety_abort_reason({"stateEstimate.roll": 50.0}, target_height_m=0.5).startswith("roll_gt_45deg")
    assert safety_abort_reason({"stateEstimate.pitch": -40.0}, target_height_m=0.5).startswith("pitch_gt_35deg")
    assert safety_abort_reason({"gyro.x": 0.0, "gyro.y": 600.0, "gyro.z": 0.0}, target_height_m=0.5).startswith("gyro_gt_500dps")
    assert safety_abort_reason({"stateEstimate.z": 0.95}, target_height_m=0.5).startswith("height_error_gt_35cm")


def test_safety_abort_can_disable_target_relative_height_guard() -> None:
    assert safety_abort_reason({"stateEstimate.z": 0.95}, target_height_m=0.5, height_error_abort_m=0.0) is None
    assert safety_abort_reason({"stateEstimate.z": 1.25}, target_height_m=0.5, height_error_abort_m=0.0).startswith("state_height_above_max")
    assert safety_abort_reason({"stateEstimate.z": 0.05}, target_height_m=0.5, height_error_abort_m=0.0).startswith("state_height_below_min")


def test_ttc_controller_uses_range_rate_input() -> None:
    from flightrl.hardware.avoidance_live import build_control_command

    model = TTCAvoidancePolicy(hidden_size=8)
    for parameter in model.parameters():
        parameter.data.zero_()
    model.net[-1].bias.data[0] = -0.4
    args = SimpleNamespace(controller="ttc-policy", max_speed_m_s=0.5, height_m=0.5, lock_height=True)

    command = build_control_command(
        model,
        RangerReading(0.6, 2.0, 2.0, 2.0, 2.0, 0.5),
        RangerReading(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        args,
    )

    assert command.vx_m_s < 0.0
    assert command.zdistance_m == 0.5
