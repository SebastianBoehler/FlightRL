#include <math.h>
#include <stdint.h>
#include "native_sixdof.h"

#define SIXDOF_ROOM_X_MIN -2.0f
#define SIXDOF_ROOM_X_MAX 2.0f
#define SIXDOF_ROOM_Y_MIN -2.0f
#define SIXDOF_ROOM_Y_MAX 2.0f
#define SIXDOF_ROOM_Z_MIN 0.0f
#define SIXDOF_ROOM_Z_MAX 2.5f
#define SIXDOF_GRAVITY 9.81f
#define SIXDOF_MASS 0.036f
#define SIXDOF_DRAG 0.10f
#define SIXDOF_RATE_TAU 0.045f
#define SIXDOF_OBS_DIM 28
#define SIXDOF_PHYS_MASS 0
#define SIXDOF_PHYS_GRAVITY 1
#define SIXDOF_PHYS_LINEAR_DRAG 2
#define SIXDOF_PHYS_RATE_TAU 3
#define SIXDOF_PHYS_THRUST_SCALE 4
#define SIXDOF_PHYS_MAX_RATE_ROLL 5
#define SIXDOF_PHYS_MAX_RATE_PITCH 6
#define SIXDOF_PHYS_MAX_RATE_YAW 7
#define SIXDOF_PHYS_MOTOR_TAU 8
#define SIXDOF_TASK_CIRCLE 3
#define SIXDOF_REWARD_PROGRESS 1
#define SIXDOF_REWARD_PROGRESS_CLEARANCE 2
#define SIXDOF_REWARD_PROGRESS_YAW_CLEARANCE 3
#define SIXDOF_REWARD_LIVE_CLEARANCE 4

static const float DEFAULT_ROOM[7] = {
    SIXDOF_ROOM_X_MIN, SIXDOF_ROOM_X_MAX, SIXDOF_ROOM_Y_MIN, SIXDOF_ROOM_Y_MAX, SIXDOF_ROOM_Z_MIN, SIXDOF_ROOM_Z_MAX, 4.0f
};
static const float DEFAULT_PHYSICS[SIXDOF_PHYSICS_DIM] = {SIXDOF_MASS, SIXDOF_GRAVITY, SIXDOF_DRAG, SIXDOF_RATE_TAU, 0.75f, 6.0f, 6.0f, 4.0f, 0.0f};
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

static const float *room_or_default(const float *room) {
    return room ? room : DEFAULT_ROOM;
}

static const float *physics_or_default(const float *physics) {
    return physics ? physics : DEFAULT_PHYSICS;
}

static float raycast_room(const float *p, const float *d, const float *room) {
    const float *r = room_or_default(room);
    float best = r[6];
    const float lows[3] = {r[0], r[2], r[4]};
    const float highs[3] = {r[1], r[3], r[5]};
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
    return clampf(best, 0.0f, r[6]);
}

static void update_ranges(const float *p, const float *q, float *ranges, const float *room) {
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
        ranges[i] = raycast_room(p, dirs[i], room);
    }
}

#include "native_sixdof_step.inc"

static float yaw_from_quat(const float *q) {
    return atan2f(2.0f * (q[0] * q[3] + q[1] * q[2]), 1.0f - 2.0f * (q[2] * q[2] + q[3] * q[3]));
}

static float wrap_angle(float value) {
    const float two_pi = 6.283185307179586f;
    while (value > 3.141592653589793f) {
        value -= two_pi;
    }
    while (value < -3.141592653589793f) {
        value += two_pi;
    }
    return value;
}

static float circle_tangent_yaw(const float *p, const float *target) {
    float rx = p[0] - target[0];
    float ry = p[1] - target[1];
    float radius = sqrtf(rx * rx + ry * ry);
    if (radius < 0.2f) {
        radius = 0.2f;
    }
    return atan2f(rx / radius, -ry / radius);
}

static float task_yaw_target(const float *p, const float *target, float target_yaw, int task_id) {
    return task_id == SIXDOF_TASK_CIRCLE ? circle_tangent_yaw(p, target) : target_yaw;
}

static float norm3(const float *values);

static float task_position_error(const float *p, const float *target, int task_id) {
    if (task_id == SIXDOF_TASK_CIRCLE) {
        float dx = p[0] - target[0];
        float dy = p[1] - target[1];
        float radius_error = sqrtf(dx * dx + dy * dy) - 0.75f;
        float z_error = p[2] - 0.65f;
        return sqrtf(radius_error * radius_error + z_error * z_error);
    }
    float err[3] = {target[0] - p[0], target[1] - p[1], target[2] - p[2]};
    return norm3(err);
}

static unsigned char in_room(const float *p, const float *room) {
    const float *r = room_or_default(room);
    const float margin = 0.03f;
    return p[0] >= r[0] + margin && p[0] <= r[1] - margin && p[1] >= r[2] + margin && p[1] <= r[3] - margin &&
        p[2] >= r[4] + margin && p[2] <= r[5] - margin;
}

static float norm3(const float *values) {
    return sqrtf(values[0] * values[0] + values[1] * values[1] + values[2] * values[2]);
}

static void assemble_one(
    const float *p,
    const float *v,
    const float *q,
    const float *rates,
    const float *ranges,
    const float *target,
    float target_yaw,
    const float *action,
    int step_count,
    int task_id,
    int reward_mode,
    float previous_error,
    const float *physics,
    float *obs,
    float *reward,
    unsigned char *terminal,
    unsigned char *truncation,
    const float *room
) {
    const float *phys = physics_or_default(physics);
    float pos_error[3] = {target[0] - p[0], target[1] - p[1], target[2] - p[2]};
    float effective_yaw = task_yaw_target(p, target, target_yaw, task_id);
    float yaw_error = wrap_angle(effective_yaw - yaw_from_quat(q));
    float min_clearance = ranges[0];
    for (int i = 1; i < 4; ++i) {
        if (ranges[i] < min_clearance) {
            min_clearance = ranges[i];
        }
    }
    float clearance_penalty = fmaxf(0.0f, 0.35f - min_clearance);
    float action_norm = sqrtf(action[0] * action[0] + action[1] * action[1] + action[2] * action[2] + action[3] * action[3]);
    float current_error = task_position_error(p, target, task_id);
    *reward = 1.0f - norm3(pos_error) - 0.15f * norm3(v) - 0.1f * fabsf(yaw_error) - 1.5f * clearance_penalty - 0.02f * action_norm;
    *terminal = in_room(p, room) ? 0 : 1;
    *truncation = step_count >= 800 ? 1 : 0;
    if (reward_mode >= SIXDOF_REWARD_PROGRESS && reward_mode <= SIXDOF_REWARD_LIVE_CLEARANCE) {
        float threshold = reward_mode == SIXDOF_REWARD_PROGRESS ? 0.25f : (reward_mode == SIXDOF_REWARD_LIVE_CLEARANCE ? 0.65f : 0.45f);
        float clearance_weight = reward_mode == SIXDOF_REWARD_PROGRESS ? 1.0f : (reward_mode == SIXDOF_REWARD_LIVE_CLEARANCE ? 5.0f : 2.5f);
        float yaw_weight = reward_mode == SIXDOF_REWARD_PROGRESS_YAW_CLEARANCE ? 0.6f : 0.1f;
        float progress_clearance = fmaxf(0.0f, threshold - min_clearance);
        float clearance_bonus = reward_mode == SIXDOF_REWARD_LIVE_CLEARANCE ? 0.15f * fminf(min_clearance, threshold) : 0.0f;
        *reward = 0.2f + 3.0f * (previous_error - current_error) - 0.05f * current_error - 0.02f * norm3(v) -
            yaw_weight * fabsf(yaw_error) - clearance_weight * progress_clearance + clearance_bonus - 0.01f * action_norm -
            ((*terminal || *truncation) ? 1.0f : 0.0f);
    }
    obs[0] = pos_error[0] / 2.0f;
    obs[1] = pos_error[1] / 2.0f;
    obs[2] = pos_error[2] / 1.5f;
    obs[3] = v[0] / 3.0f;
    obs[4] = v[1] / 3.0f;
    obs[5] = v[2] / 3.0f;
    for (int i = 0; i < 4; ++i) {
        obs[6 + i] = q[i];
    }
    obs[10] = rates[0] / phys[SIXDOF_PHYS_MAX_RATE_ROLL];
    obs[11] = rates[1] / phys[SIXDOF_PHYS_MAX_RATE_PITCH];
    obs[12] = rates[2] / phys[SIXDOF_PHYS_MAX_RATE_YAW];
    obs[13] = target[0] / 2.0f;
    obs[14] = target[1] / 2.0f;
    obs[15] = target[2] / 2.5f;
    obs[16] = sinf(yaw_error);
    obs[17] = cosf(yaw_error);
    for (int i = 0; i < 6; ++i) {
        obs[18 + i] = ranges[i] / room_or_default(room)[6];
    }
    for (int i = 0; i < 4; ++i) {
        obs[24 + i] = action[i];
    }
}

void flightrl_sixdof_step_batch(float *position, float *velocity, float *quaternion, float *body_rates, float *ranges, float *thrust_state, const float *actions, const float *physics, const float *room, int num_envs, float dt) {
    for (int env = 0; env < num_envs; ++env) {
        step_one(position + env * 3, velocity + env * 3, quaternion + env * 4, body_rates + env * 3, ranges + env * 6,
            thrust_state + env, actions + env * 4, physics + env * SIXDOF_PHYSICS_DIM, room, dt);
    }
}

void flightrl_sixdof_step_env_batch(
    float *position, float *velocity,
    float *quaternion,
    float *body_rates,
    float *ranges,
    float *thrust_state,
    const float *physics,
    float *target_position,
    float *target_yaw,
    float *previous_action,
    int *step_count,
    const float *actions,
    float *observations,
    float *rewards,
    unsigned char *terminals,
    unsigned char *truncations,
    const float *room,
    int num_envs,
    float dt
) {
    for (int env = 0; env < num_envs; ++env) {
        float *action = previous_action + env * 4;
        for (int i = 0; i < 4; ++i) {
            action[i] = clampf(actions[env * 4 + i], -1.0f, 1.0f);
        }
        const float *phys = physics + env * SIXDOF_PHYSICS_DIM;
        step_one(position + env * 3, velocity + env * 3, quaternion + env * 4, body_rates + env * 3, ranges + env * 6,
            thrust_state + env, action, phys, room, dt);
        step_count[env] += 1;
        assemble_one(
            position + env * 3,
            velocity + env * 3,
            quaternion + env * 4,
            body_rates + env * 3,
            ranges + env * 6,
            target_position + env * 3,
            target_yaw[env],
            action,
            step_count[env],
            0,
            0,
            0.0f,
            phys,
            observations + env * SIXDOF_OBS_DIM,
            rewards + env,
            terminals + env,
            truncations + env,
            room
        );
    }
}
#include "native_sixdof_context.inc"
