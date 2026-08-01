#ifndef FLIGHTRL_NATIVE_SIXDOF_SETPOINT_H
#define FLIGHTRL_NATIVE_SIXDOF_SETPOINT_H

void flightrl_sixdof_setpoint_actions_batch(
    const float *velocity,
    const float *quaternion,
    const float *setpoints,
    const float *physics,
    float *low_level_actions,
    int num_envs,
    float max_horizontal_speed,
    float max_vertical_speed,
    float velocity_gain,
    float attitude_gain,
    float vertical_gain
);

void flightrl_sixdof_waypoint_residual_actions_batch(
    const float *position,
    const float *velocity,
    const float *quaternion,
    const float *target_position,
    const float *target_yaw,
    const float *residual_setpoints,
    const float *physics,
    float *low_level_actions,
    int num_envs,
    float max_horizontal_speed,
    float max_vertical_speed,
    float velocity_gain,
    float attitude_gain,
    float vertical_gain,
    float residual_scale,
    float slowdown_distance
);

float flightrl_sixdof_avoidance_alignment(
    const float *position,
    const float *quaternion,
    const float *obstacle,
    const float *residual_setpoint
);

float flightrl_sixdof_clearance_deficit(
    const float *position,
    const float *obstacle
);

#endif
