#include <math.h>

#include "native_sixdof_setpoint.h"

#define SETPOINT_PHYS_GRAVITY 1
#define SETPOINT_PHYS_MAX_RATE_ROLL 5
#define SETPOINT_PHYS_MAX_RATE_PITCH 6

static float setpoint_clamp(float value, float lo, float hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

static float setpoint_yaw(const float *q) {
    return atan2f(2.0f * (q[0] * q[3] + q[1] * q[2]), 1.0f - 2.0f * (q[2] * q[2] + q[3] * q[3]));
}

static void setpoint_roll_pitch(const float *q, float *roll, float *pitch) {
    *roll = atan2f(
        2.0f * (q[0] * q[1] + q[2] * q[3]),
        1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2])
    );
    *pitch = asinf(setpoint_clamp(2.0f * (q[0] * q[2] - q[3] * q[1]), -1.0f, 1.0f));
}

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
) {
    for (int env = 0; env < num_envs; ++env) {
        const float *q = quaternion + env * 4;
        const float *v = velocity + env * 3;
        const float *command = setpoints + env * 4;
        const float *phys = physics + env * 9;
        float *action = low_level_actions + env * 4;
        float yaw = setpoint_yaw(q);
        float cosine = cosf(yaw);
        float sine = sinf(yaw);

        float target_body_x = setpoint_clamp(command[0], -1.0f, 1.0f) * max_horizontal_speed;
        float target_body_y = setpoint_clamp(command[1], -1.0f, 1.0f) * max_horizontal_speed;
        float target_world_z = setpoint_clamp(command[2], -1.0f, 1.0f) * max_vertical_speed;
        float current_body_x = cosine * v[0] + sine * v[1];
        float current_body_y = -sine * v[0] + cosine * v[1];
        float error_body_x = target_body_x - current_body_x;
        float error_body_y = target_body_y - current_body_y;

        float gravity = fmaxf(phys[SETPOINT_PHYS_GRAVITY], 1.0e-6f);
        float desired_pitch = setpoint_clamp(velocity_gain * error_body_x / gravity, -0.25f, 0.25f);
        float desired_roll = setpoint_clamp(-velocity_gain * error_body_y / gravity, -0.25f, 0.25f);
        float roll;
        float pitch;
        setpoint_roll_pitch(q, &roll, &pitch);

        action[0] = setpoint_clamp(vertical_gain * (target_world_z - v[2]) / gravity, -1.0f, 1.0f);
        action[1] = setpoint_clamp(
            attitude_gain * (desired_roll - roll) / fmaxf(phys[SETPOINT_PHYS_MAX_RATE_ROLL], 1.0e-6f),
            -1.0f,
            1.0f
        );
        action[2] = setpoint_clamp(
            attitude_gain * (desired_pitch - pitch) / fmaxf(phys[SETPOINT_PHYS_MAX_RATE_PITCH], 1.0e-6f),
            -1.0f,
            1.0f
        );
        action[3] = setpoint_clamp(command[3], -1.0f, 1.0f);
    }
}

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
) {
    for (int env = 0; env < num_envs; ++env) {
        const float *p = position + env * 3;
        const float *q = quaternion + env * 4;
        const float *target = target_position + env * 3;
        const float *residual = residual_setpoints + env * 4;
        float dx = target[0] - p[0];
        float dy = target[1] - p[1];
        float dz = target[2] - p[2];
        float distance = sqrtf(dx * dx + dy * dy + dz * dz);
        float scale = setpoint_clamp(distance / fmaxf(slowdown_distance, 1.0e-3f), 0.0f, 1.0f);
        float inverse_distance = distance > 1.0e-6f ? 1.0f / distance : 0.0f;
        float yaw = setpoint_yaw(q);
        float cosine = cosf(yaw);
        float sine = sinf(yaw);
        float command[4] = {
            scale * (cosine * dx + sine * dy) * inverse_distance + residual_scale * residual[0],
            scale * (-sine * dx + cosine * dy) * inverse_distance + residual_scale * residual[1],
            scale * dz * inverse_distance + residual_scale * residual[2],
            sinf(target_yaw[env] - yaw) + residual_scale * residual[3],
        };
        flightrl_sixdof_setpoint_actions_batch(
            velocity + env * 3,
            q,
            command,
            physics + env * 9,
            low_level_actions + env * 4,
            1,
            max_horizontal_speed,
            max_vertical_speed,
            velocity_gain,
            attitude_gain,
            vertical_gain
        );
    }
}

float flightrl_sixdof_avoidance_alignment(
    const float *position,
    const float *quaternion,
    const float *obstacle,
    const float *residual_setpoint
) {
    if (obstacle[0] > 5.0f) {
        return 0.0f;
    }
    float dx = 0.5f * (obstacle[0] + obstacle[1]) - position[0];
    float dy = 0.5f * (obstacle[2] + obstacle[3]) - position[1];
    float yaw = setpoint_yaw(quaternion);
    float body_x = cosf(yaw) * dx + sinf(yaw) * dy;
    float body_y = -sinf(yaw) * dx + cosf(yaw) * dy;
    if (body_x <= 0.0f || body_x >= 1.5f) {
        return 0.0f;
    }
    float approach = 1.0f - body_x / 1.5f;
    float preferred_side = body_y >= 0.0f ? -1.0f : 1.0f;
    return approach * preferred_side * residual_setpoint[1];
}
