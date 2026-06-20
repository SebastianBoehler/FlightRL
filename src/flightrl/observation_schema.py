from __future__ import annotations

POSITION_DIM = 2
VELOCITY_DIM = 2
ATTITUDE_DIM = 2
ANGULAR_VELOCITY_DIM = 1
TARGET_VECTOR_DIM = 2
HEALTH_DIM = 1
IDEAL_STATE_DIM = 6
NOISY_STATE_DIM = 6
IMU_DIM = 3
RANGE_SENSOR_DIM = 5
RANGE_RATE_SENSOR_DIM = 6
TTC_SENSOR_DIM = 2
VISION_SENSOR_DIM = 0

CRAZYFLIE_TELEMETRY_BASE_DIM = 28

OBSERVATION_FLAG_BITS = {
    "include_position": 1 << 0,
    "include_velocity": 1 << 1,
    "include_attitude": 1 << 2,
    "include_angular_velocity": 1 << 3,
    "include_target_vector": 1 << 4,
    "include_previous_action": 1 << 5,
    "include_health": 1 << 6,
    "include_ideal_state": 1 << 7,
    "include_noisy_state": 1 << 8,
    "include_imu": 1 << 9,
    "include_range_sensor": 1 << 10,
    "include_vision_sensor": 1 << 11,
    "include_crazyflie_telemetry": 1 << 12,
    "include_range_rate_sensor": 1 << 13,
    "include_ttc_sensor": 1 << 14,
}
