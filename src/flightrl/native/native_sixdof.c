#include <math.h>

#include "native_sixdof.h"

#define SIXDOF_ROOM_X_MIN -2.0f
#define SIXDOF_ROOM_X_MAX 2.0f
#define SIXDOF_ROOM_Y_MIN -2.0f
#define SIXDOF_ROOM_Y_MAX 2.0f
#define SIXDOF_ROOM_Z_MIN 0.0f
#define SIXDOF_ROOM_Z_MAX 2.5f
#define SIXDOF_MAX_RANGE 4.0f
#define SIXDOF_GRAVITY 9.81f
#define SIXDOF_MASS 0.036f
#define SIXDOF_DRAG 0.10f
#define SIXDOF_RATE_TAU 0.045f

static float clampf(float value, float lo, float hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

static void normalize_quat(float *q) {
    float norm = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (norm < 1.0e-8f) {
        q[0] = 1.0f;
        q[1] = q[2] = q[3] = 0.0f;
        return;
    }
    float inv = 1.0f / norm;
    q[0] *= inv;
    q[1] *= inv;
    q[2] *= inv;
    q[3] *= inv;
}

static void quat_matrix(const float *q, float r[9]) {
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

static float raycast_room(const float *p, const float *d) {
    float best = SIXDOF_MAX_RANGE;
    const float lows[3] = {SIXDOF_ROOM_X_MIN, SIXDOF_ROOM_Y_MIN, SIXDOF_ROOM_Z_MIN};
    const float highs[3] = {SIXDOF_ROOM_X_MAX, SIXDOF_ROOM_Y_MAX, SIXDOF_ROOM_Z_MAX};
    for (int axis = 0; axis < 3; ++axis) {
        if (fabsf(d[axis]) < 1.0e-6f) {
            continue;
        }
        float plane = d[axis] > 0.0f ? highs[axis] : lows[axis];
        float t = (plane - p[axis]) / d[axis];
        if (t > 1.0e-6f && t < best) {
            best = t;
        }
    }
    return clampf(best, 0.0f, SIXDOF_MAX_RANGE);
}

static void update_ranges(const float *p, const float *q, float *ranges) {
    float r[9];
    quat_matrix(q, r);
    float dirs[6][3] = {
        {r[0], r[3], r[6]},
        {-r[0], -r[3], -r[6]},
        {r[1], r[4], r[7]},
        {-r[1], -r[4], -r[7]},
        {r[2], r[5], r[8]},
        {-r[2], -r[5], -r[8]},
    };
    for (int i = 0; i < 6; ++i) {
        ranges[i] = raycast_room(p, dirs[i]);
    }
}

static void step_one(float *p, float *v, float *q, float *rates, float *ranges, const float *action, float dt) {
    const float max_rate[3] = {6.0f, 6.0f, 4.0f};
    float alpha = dt / (SIXDOF_RATE_TAU + dt);
    float thrust = SIXDOF_MASS * SIXDOF_GRAVITY * (1.0f + 0.75f * clampf(action[0], -1.0f, 1.0f));

    for (int i = 0; i < 3; ++i) {
        float target = clampf(action[i + 1], -1.0f, 1.0f) * max_rate[i];
        rates[i] += alpha * (target - rates[i]);
    }

    float omega[4] = {0.0f, rates[0], rates[1], rates[2]};
    float qdot[4] = {
        -0.5f * (q[1] * omega[1] + q[2] * omega[2] + q[3] * omega[3]),
        0.5f * (q[0] * omega[1] + q[2] * omega[3] - q[3] * omega[2]),
        0.5f * (q[0] * omega[2] - q[1] * omega[3] + q[3] * omega[1]),
        0.5f * (q[0] * omega[3] + q[1] * omega[2] - q[2] * omega[1]),
    };
    for (int i = 0; i < 4; ++i) {
        q[i] += qdot[i] * dt;
    }
    normalize_quat(q);

    float r[9];
    quat_matrix(q, r);
    float accel[3] = {r[2] * thrust / SIXDOF_MASS, r[5] * thrust / SIXDOF_MASS, r[8] * thrust / SIXDOF_MASS - SIXDOF_GRAVITY};
    for (int i = 0; i < 3; ++i) {
        accel[i] -= SIXDOF_DRAG * v[i];
        v[i] += accel[i] * dt;
        p[i] += v[i] * dt;
    }
    update_ranges(p, q, ranges);
}

void flightrl_sixdof_step_batch(float *position, float *velocity, float *quaternion, float *body_rates, float *ranges, const float *actions, int num_envs, float dt) {
    for (int env = 0; env < num_envs; ++env) {
        step_one(position + env * 3, velocity + env * 3, quaternion + env * 4, body_rates + env * 3, ranges + env * 6, actions + env * 4, dt);
    }
}
