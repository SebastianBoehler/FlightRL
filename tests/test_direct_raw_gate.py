from __future__ import annotations

from flightrl.hardware.direct_raw_gate import DirectRawGateThresholds, evaluate_direct_raw_replay


def row(**overrides: float) -> dict[str, float]:
    base = {
        "sys.isTumbled": 0.0,
        "sys.canfly": 1.0,
        "stabilizer.roll": 1.0,
        "stabilizer.pitch": 1.0,
        "stateEstimate.z": 0.5,
        "stateEstimate.vx": 0.0,
        "stateEstimate.vy": 0.0,
        "stateEstimate.vz": 0.0,
        "range.zrange": 500.0,
        "range.front": 800.0,
        "range.back": 900.0,
        "range.left": 700.0,
        "range.right": 600.0,
        "action_thrust": 0.05,
        "action_roll_rate": 0.05,
        "action_pitch_rate": -0.05,
        "action_yaw_rate": 0.02,
        "thrust_percent": 50.0,
        "roll_rate_deg_s": 20.0,
        "pitch_rate_deg_s": -20.0,
        "commander_pitch_rate_deg_s": 20.0,
        "yaw_rate_deg_s": 5.0,
    }
    base.update(overrides)
    return base


def test_direct_raw_gate_passes_moderate_safe_rows() -> None:
    report = evaluate_direct_raw_replay([row() for _ in range(4)], DirectRawGateThresholds(min_safe_rows=4))

    assert report["passed"] is True


def test_direct_raw_gate_fails_saturated_close_rows() -> None:
    rows = [
        row(**{"range.back": 120.0, "action_pitch_rate": 1.0, "pitch_rate_deg_s": 343.0, "commander_pitch_rate_deg_s": -343.0})
        for _ in range(4)
    ]
    report = evaluate_direct_raw_replay(rows, DirectRawGateThresholds(min_safe_rows=4))

    assert report["passed"] is False
    assert "close_action_saturation" in report["failures"]


def test_direct_raw_gate_excludes_high_speed_crash_rows() -> None:
    rows = [row(**{"stateEstimate.vx": 3.4}) for _ in range(4)]
    report = evaluate_direct_raw_replay(rows, DirectRawGateThresholds(min_safe_rows=4))

    assert report["passed"] is False
    assert report["safe_rows"] == 0
    assert "too_few_safe_rows" in report["failures"]


def test_direct_raw_gate_fails_source_crash_after_safe_rows() -> None:
    rows = [row() for _ in range(4)] + [
        row(**{"sys.isTumbled": 1.0, "range.front": 24.0, "stateEstimate.roll": -120.0}) for _ in range(4)
    ]

    report = evaluate_direct_raw_replay(rows, DirectRawGateThresholds(min_safe_rows=4))

    assert report["passed"] is False
    assert report["safe_rows"] == 4
    assert "source_tumble" in report["failures"]
    assert "source_near_contact" in report["failures"]
    assert "source_extreme_tilt" in report["failures"]


def test_direct_raw_gate_can_score_commands_without_source_failure() -> None:
    rows = [row() for _ in range(4)] + [
        row(**{"sys.isTumbled": 1.0, "range.front": 24.0, "stateEstimate.roll": -120.0}) for _ in range(4)
    ]

    report = evaluate_direct_raw_replay(rows, DirectRawGateThresholds(min_safe_rows=4, require_source_health=False))

    assert report["passed"] is True
    assert report["source_tumble_rows"] == 4
    assert "source_tumble" not in report["failures"]


def test_direct_raw_gate_fails_without_explicit_commander_pitch_sign() -> None:
    rows = [row() for _ in range(4)]
    for item in rows:
        item.pop("commander_pitch_rate_deg_s")

    report = evaluate_direct_raw_replay(rows, DirectRawGateThresholds(min_safe_rows=4))

    assert report["passed"] is False
    assert "missing_commander_pitch_sign" in report["failures"]


def test_direct_raw_gate_fails_commander_pitch_sign_mismatch() -> None:
    rows = [row(**{"pitch_rate_deg_s": 30.0, "commander_pitch_rate_deg_s": 30.0}) for _ in range(4)]

    report = evaluate_direct_raw_replay(rows, DirectRawGateThresholds(min_safe_rows=4))

    assert report["passed"] is False
    assert "commander_pitch_sign_mismatch" in report["failures"]


def test_direct_raw_gate_fails_precontact_open_space_drift() -> None:
    rows = [row(**{"range.front": 700.0, "stateEstimate.vx": 0.55}) for _ in range(4)]

    report = evaluate_direct_raw_replay(rows, DirectRawGateThresholds(min_safe_rows=4))

    assert report["passed"] is False
    assert report["source"]["precontact_high_speed_rows"] == 4
    assert "source_precontact_drift" in report["failures"]
