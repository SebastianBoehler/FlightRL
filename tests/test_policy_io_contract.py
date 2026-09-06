from __future__ import annotations

import pytest

from flightrl.policy_io_contract import (
    ActionMode,
    ActionSpec,
    InputSignal,
    PolicyIOContract,
    SignalDType,
    SignalEncoding,
    SignalRole,
    compile_policy_io_contract,
)


def _raw_contract(*, rotors: int = 4) -> PolicyIOContract:
    return PolicyIOContract(
        inputs=(
            InputSignal(
                name="camera_front",
                role=SignalRole.SENSOR,
                dtype=SignalDType.UINT8,
                shape=(64, 96),
                unit="gray8",
                frame="camera_front",
                sample_rate_hz=30.0,
                encoding=SignalEncoding.RAW,
                normalization_scale=1.0 / 255.0,
            ),
            InputSignal(
                name="imu_gyro",
                role=SignalRole.SENSOR,
                dtype=SignalDType.FLOAT32,
                shape=(3,),
                unit="rad_s",
                frame="body_flu",
                sample_rate_hz=500.0,
                encoding=SignalEncoding.CALIBRATED,
            ),
            InputSignal(
                name="motor_rpm",
                role=SignalRole.VEHICLE,
                dtype=SignalDType.FLOAT32,
                shape=(rotors,),
                unit="rpm",
                frame="actuator_order",
                sample_rate_hz=500.0,
                encoding=SignalEncoding.CALIBRATED,
                normalization_scale=1.0 / 30_000.0,
            ),
            InputSignal(
                name="esc_temperature",
                role=SignalRole.VEHICLE,
                dtype=SignalDType.FLOAT32,
                shape=(rotors,),
                unit="celsius",
                frame="actuator_order",
                sample_rate_hz=50.0,
                encoding=SignalEncoding.CALIBRATED,
                normalization_scale=1.0 / 100.0,
            ),
            InputSignal(
                name="goal_conditioning",
                role=SignalRole.GOAL,
                dtype=SignalDType.FLOAT32,
                shape=(8,),
                unit="typed_goal",
                frame="mission",
                sample_rate_hz=10.0,
                encoding=SignalEncoding.CALIBRATED,
            ),
        ),
        action=ActionSpec.direct_motor_thrust(rotor_count=rotors, rate_hz=500.0),
    )


def test_compiles_contiguous_raw_to_direct_motor_contract() -> None:
    compiled = compile_policy_io_contract(_raw_contract())

    assert compiled["schema"] == "flightrl.policy_io.v1"
    assert compiled["deployment_authority"] is False
    assert compiled["observation"]["feature_engineering"] == "none"
    assert [signal["name"] for signal in compiled["observation"]["signals"]] == [
        "camera_front",
        "imu_gyro",
        "motor_rpm",
        "esc_temperature",
        "goal_conditioning",
    ]
    offsets = [signal["byte_offset"] for signal in compiled["observation"]["signals"]]
    assert offsets == [0, 64 * 96, 64 * 96 + 12, 64 * 96 + 28, 64 * 96 + 44]
    assert compiled["action"]["mode"] == "direct_motor_thrust"
    assert compiled["action"]["fields"] == [
        "motor_0",
        "motor_1",
        "motor_2",
        "motor_3",
    ]
    assert compiled["action"]["normalized_bounds"] == [0.0, 1.0]
    assert compiled["action"]["field_binding"] == "embodiment_actuator_order"
    assert compiled["action"]["applied_action_feedback_required"] is True
    assert compiled["observation"]["temporal_metadata"] == {
        "capture_time": "required_monotonic_uint64_us",
        "sequence": "required_uint32",
        "validity": "required_explicit_mask",
        "missing_data": "never_zero_fill_as_valid",
    }


def test_same_observation_contract_supports_different_rotor_topologies() -> None:
    quad = compile_policy_io_contract(_raw_contract(rotors=4))
    hexacopter = compile_policy_io_contract(_raw_contract(rotors=6))

    assert quad["action"]["width"] == 4
    assert hexacopter["action"]["width"] == 6
    assert hexacopter["action"]["fields"][-1] == "motor_5"
    assert hexacopter["observation"]["signals"][2]["shape"] == [6]


def test_setpoint_and_direct_control_modes_are_explicitly_distinct() -> None:
    direct = ActionSpec.direct_motor_thrust(rotor_count=4, rate_hz=500.0)
    rates = ActionSpec.body_rates_and_thrust(rate_hz=250.0)
    velocity = ActionSpec.velocity_and_yaw_rate(rate_hz=50.0)

    assert direct.mode is ActionMode.DIRECT_MOTOR_THRUST
    assert rates.fields == ("roll_rate", "pitch_rate", "yaw_rate", "collective_thrust")
    assert velocity.fields == ("vx", "vy", "vz", "yaw_rate")


def test_contract_rejects_ambiguous_or_engineered_inputs() -> None:
    signal = InputSignal(
        name="camera_front",
        role=SignalRole.SENSOR,
        dtype=SignalDType.UINT8,
        shape=(64, 96),
        unit="gray8",
        frame="camera_front",
        sample_rate_hz=30.0,
        encoding=SignalEncoding.RAW,
        normalization_scale=1.0 / 255.0,
    )
    with pytest.raises(ValueError, match="unique"):
        PolicyIOContract(
            inputs=(signal, signal),
            action=ActionSpec.velocity_and_yaw_rate(rate_hz=50.0),
        )
    with pytest.raises(ValueError, match="identity or affine calibration"):
        InputSignal(
            name="distance_to_goal",
            role=SignalRole.SENSOR,
            dtype=SignalDType.FLOAT32,
            shape=(1,),
            unit="m",
            frame="world",
            sample_rate_hz=50.0,
            encoding=SignalEncoding.DERIVED,
        )
    assert "MISSION" not in SignalRole.__members__


def test_contract_identity_is_deterministic_and_bound() -> None:
    first = compile_policy_io_contract(_raw_contract())
    second = compile_policy_io_contract(_raw_contract())

    assert first == second
    assert len(first["sha256"]) == 64
