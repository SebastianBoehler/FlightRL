#include <math.h>

#include "native_door_proprio.h"

static float proprio_clamp(float value, float lo, float hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

static void quaternion_matrix(const float *q, float r[9]) {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    r[0] = 1.0f - 2.0f * (y * y + z * z);
    r[1] = 2.0f * (x * y - z * w);
    r[2] = 2.0f * (x * z + y * w);
    r[3] = 2.0f * (x * y + z * w);
    r[4] = 1.0f - 2.0f * (x * x + z * z);
    r[5] = 2.0f * (y * z - x * w);
    r[6] = 2.0f * (x * z - y * w);
    r[7] = 2.0f * (y * z + x * w);
    r[8] = 1.0f - 2.0f * (x * x + y * y);
}

float flightrl_door_yaw(const float *q) {
    return atan2f(
        2.0f * (q[0] * q[3] + q[1] * q[2]),
        1.0f - 2.0f * (q[2] * q[2] + q[3] * q[3])
    );
}

void flightrl_door_proprioception(
    const float *position,
    const float *velocity,
    const float *quaternion,
    const float *body_rates,
    const float *room,
    const float *origin_position,
    float origin_yaw,
    const float *previous_action,
    float *output
) {
    float rotation[9];
    quaternion_matrix(quaternion, rotation);
    output[0] = proprio_clamp(
        rotation[0] * velocity[0]
            + rotation[3] * velocity[1]
            + rotation[6] * velocity[2],
        -1.0f,
        1.0f
    );
    output[1] = proprio_clamp(
        rotation[1] * velocity[0]
            + rotation[4] * velocity[1]
            + rotation[7] * velocity[2],
        -1.0f,
        1.0f
    );
    output[2] = proprio_clamp(
        2.0f * (
            rotation[2] * velocity[0]
                + rotation[5] * velocity[1]
                + rotation[8] * velocity[2]
        ),
        -1.0f,
        1.0f
    );
    output[3] = proprio_clamp(body_rates[0] / 6.0f, -1.0f, 1.0f);
    output[4] = proprio_clamp(body_rates[1] / 6.0f, -1.0f, 1.0f);
    output[5] = proprio_clamp(body_rates[2] / 4.0f, -1.0f, 1.0f);
    output[6] = rotation[6];
    output[7] = rotation[7];
    output[8] = rotation[8];
    output[9] = proprio_clamp(
        (position[2] - room[4]) / fmaxf(room[5] - room[4], 1.0e-3f),
        0.0f,
        1.0f
    );
    float dx = position[0] - origin_position[0];
    float dy = position[1] - origin_position[1];
    float cosine = cosf(origin_yaw);
    float sine = sinf(origin_yaw);
    output[10] = proprio_clamp((cosine * dx + sine * dy) / 4.0f, -1.0f, 1.0f);
    output[11] = proprio_clamp((-sine * dx + cosine * dy) / 4.0f, -1.0f, 1.0f);
    output[12] = proprio_clamp(
        (position[2] - origin_position[2]) / 2.0f,
        -1.0f,
        1.0f
    );
    float relative_yaw = flightrl_door_yaw(quaternion) - origin_yaw;
    output[13] = sinf(relative_yaw);
    output[14] = cosf(relative_yaw);
    output[15] = previous_action[0];
    output[16] = previous_action[1];
}
