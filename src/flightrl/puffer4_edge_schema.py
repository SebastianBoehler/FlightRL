from __future__ import annotations


EDGE_WIDTH = 64
EDGE_HEIGHT = 48
EDGE_FRAME_PIXELS = EDGE_WIDTH * EDGE_HEIGHT
EDGE_TELEMETRY_DIM = 19
EDGE_TARGET_VOCABULARY = ("door", "monitor", "sink")
EDGE_MISSION_TOKEN_COUNT = len(EDGE_TARGET_VOCABULARY)
EDGE_ACTION_DIM = 4
EDGE_OBSERVATION_DIM = (
    EDGE_FRAME_PIXELS + EDGE_TELEMETRY_DIM + EDGE_MISSION_TOKEN_COUNT
)
EDGE_POLICY_CONTRACT_ID = "aideck-navigation-policy-v3"

TELEMETRY_SPECS = (
    ("body_vx", "m/s", 1.0, (-1.0, 1.0), "body_flu_x_forward", "body_vx_m_s / 1.0"),
    ("body_vy", "m/s", 1.0, (-1.0, 1.0), "body_flu_y_left", "body_vy_m_s / 1.0"),
    ("body_vz", "m/s", 0.5, (-1.0, 1.0), "body_flu_z_up", "body_vz_m_s / 0.5"),
    ("body_rate_x", "rad/s", 6.0, (-1.0, 1.0), "body_flu_right_hand_x", "gyro_x_rad_s / 6.0"),
    ("body_rate_y", "rad/s", 6.0, (-1.0, 1.0), "body_flu_right_hand_y", "gyro_y_rad_s / 6.0"),
    ("body_rate_z", "rad/s", 4.0, (-1.0, 1.0), "body_flu_right_hand_z", "gyro_z_rad_s / 4.0"),
    ("body_up_x", "1", 1.0, (-1.0, 1.0), "world_up_expressed_in_body_flu", "world_up_dot_body_x"),
    ("body_up_y", "1", 1.0, (-1.0, 1.0), "world_up_expressed_in_body_flu", "world_up_dot_body_y"),
    ("body_up_z", "1", 1.0, (-1.0, 1.0), "world_up_expressed_in_body_flu", "world_up_dot_body_z"),
    ("altitude_fraction", "m", 2.5, (0.0, 1.0), "world_up_from_takeoff_origin", "altitude_m / 2.5"),
    ("origin_forward_displacement", "m", 4.0, (-1.0, 1.0), "mission_start_yaw_frame_x_forward", "origin_forward_m / 4.0"),
    ("origin_left_displacement", "m", 4.0, (-1.0, 1.0), "mission_start_yaw_frame_y_left", "origin_left_m / 4.0"),
    ("origin_vertical_displacement", "m", 2.0, (-1.0, 1.0), "world_up_from_mission_start", "origin_vertical_m / 2.0"),
    ("sin_relative_yaw", "1", 1.0, (-1.0, 1.0), "world_up_from_mission_start_yaw", "sin(current_yaw - mission_start_yaw)"),
    ("cos_relative_yaw", "1", 1.0, (-1.0, 1.0), "world_up_from_mission_start_yaw", "cos(current_yaw - mission_start_yaw)"),
    ("previous_vx", "m/s", 0.25, (-1.0, 1.0), "previous_step_body_flu_x_forward", "last_stm32_applied_vx_m_s / 0.25"),
    ("previous_vy", "m/s", 0.25, (-1.0, 1.0), "previous_step_body_flu_y_left", "last_stm32_applied_vy_m_s / 0.25"),
    ("previous_vz", "m/s", 0.15, (-1.0, 1.0), "world_up", "last_stm32_applied_vz_m_s / 0.15"),
    ("previous_yaw_rate", "deg/s", 45.0, (-1.0, 1.0), "world_up_positive_left", "last_stm32_applied_yaw_rate_deg_s / 45.0"),
)

ACTION_SPECS = (
    ("vx", "m/s", 0.25, "body_flu_x_forward"),
    ("vy", "m/s", 0.25, "body_flu_y_left"),
    ("vz", "m/s", 0.15, "world_up"),
    ("yaw_rate", "deg/s", 45.0, "world_up_positive_left"),
)

EDGE_TELEMETRY_BOUNDS = tuple(spec[3] for spec in TELEMETRY_SPECS)
