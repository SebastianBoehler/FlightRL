"""Bind this actor to the existing robot/sensor/action contract vocabulary."""

from flightrl.policy_io_contract import (
    ActionMode,
    ActionSpec,
    InputSignal,
    PolicyIOContract,
    SignalDType as D,
    SignalEncoding as E,
    SignalRole as R,
    compile_policy_io_contract,
)


def contract():
    return compile_policy_io_contract(
        PolicyIOContract(
            inputs=(
                InputSignal(
                    "rgb",
                    R.SENSOR,
                    D.UINT8,
                    (48, 64, 3),
                    "intensity",
                    "camera",
                    10,
                    E.RAW,
                    1 / 255,
                ),
                InputSignal(
                    "depth",
                    R.SENSOR,
                    D.FLOAT32,
                    (48, 64),
                    "m",
                    "camera_ray",
                    10,
                    E.CALIBRATED,
                    1 / 8,
                ),
                InputSignal(
                    "body_velocity",
                    R.VEHICLE,
                    D.FLOAT32,
                    (3,),
                    "m/s",
                    "body",
                    10,
                    E.CALIBRATED,
                ),
                InputSignal(
                    "gravity_direction",
                    R.VEHICLE,
                    D.FLOAT32,
                    (3,),
                    "unit_vector",
                    "body",
                    10,
                    E.CALIBRATED,
                ),
                InputSignal(
                    "gyro", R.SENSOR, D.FLOAT32, (3,), "rad/s", "body", 10, E.CALIBRATED
                ),
                InputSignal(
                    "role",
                    R.GOAL,
                    D.FLOAT32,
                    (3,),
                    "one_hot",
                    "mission",
                    10,
                    E.DISCRETE,
                ),
                InputSignal(
                    "peer_reports",
                    R.NEIGHBOR,
                    D.FLOAT32,
                    (4,),
                    "detection_and_validity_bits",
                    "sender",
                    10,
                    E.DISCRETE,
                ),
                InputSignal(
                    "peer_age",
                    R.NEIGHBOR,
                    D.FLOAT32,
                    (2,),
                    "s",
                    "sender_capture_time",
                    10,
                    E.CALIBRATED,
                ),
            ),
            action=ActionSpec(
                ActionMode.BODY_RATES_THRUST,
                ("collective_thrust", "roll_rate", "pitch_rate", "yaw_rate"),
                10,
                (-1.0, 1.0),
            ),
        )
    )
