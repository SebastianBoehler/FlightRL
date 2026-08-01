#include <math.h>

#include "native_door_teacher.h"

#define TEACHER_PI 3.14159265358979323846f
#define SEARCH_SWEEP_RAD (2.0f * TEACHER_PI - 1.4312f)

static float teacher_yaw(const float *q) {
    return atan2f(
        2.0f * (q[0] * q[3] + q[1] * q[2]),
        1.0f - 2.0f * (q[2] * q[2] + q[3] * q[3])
    );
}

static float wrap_teacher_angle(float value) {
    while (value > TEACHER_PI) value -= 2.0f * TEACHER_PI;
    while (value < -TEACHER_PI) value += 2.0f * TEACHER_PI;
    return value;
}

static void command_toward(
    const float *position,
    const float *quaternion,
    const float *goal,
    float forward,
    float max_yawrate_deg_s,
    float *action
) {
    float error = wrap_teacher_angle(
        atan2f(goal[1] - position[1], goal[0] - position[0])
        - teacher_yaw(quaternion)
    );
    action[0] = fabsf(error) < 0.28f ? forward : 0.0f;
    action[1] = clampf(
        error / (max_yawrate_deg_s * TEACHER_PI / 180.0f),
        -1.0f,
        1.0f
    );
}

void flightrl_door_teacher_advance(
    const float *position,
    const float *quaternion,
    FlightRLDoorScene *scene
) {
    float yaw = teacher_yaw(quaternion);
    if (scene->target_observed) {
        scene->search_yaw = yaw;
        return;
    }
    if (scene->search_phase != 1) {
        scene->search_yaw_progress += fabsf(
            wrap_teacher_angle(yaw - scene->search_yaw)
        );
        if (scene->search_yaw_progress >= SEARCH_SWEEP_RAD) {
            scene->search_phase = 1;
            scene->search_yaw_progress = 0.0f;
        }
    } else {
        float dx = scene->coverage[0] - position[0];
        float dy = scene->coverage[1] - position[1];
        if (sqrtf(dx * dx + dy * dy) <= 0.18f) {
            scene->search_phase = 2;
            scene->search_yaw_progress = 0.0f;
        }
    }
    scene->search_yaw = yaw;
}

void flightrl_door_teacher_action(
    const float *position,
    const float *quaternion,
    FlightRLDoorScene *scene,
    float max_yawrate_deg_s,
    float *action
) {
    if (!scene->target_observed) {
        if (scene->search_phase != 1) {
            action[0] = 0.0f;
            action[1] = 1.0f;
        } else {
            command_toward(
                position,
                quaternion,
                scene->coverage,
                0.45f,
                max_yawrate_deg_s,
                action
            );
        }
        return;
    }
    float goal[2] = {scene->target[0], scene->target[1]};
    if (scene->detour_active) {
        float dx = scene->detour[0] - position[0];
        float dy = scene->detour[1] - position[1];
        if (sqrtf(dx * dx + dy * dy) <= 0.22f) {
            scene->detour_active = 0;
        } else {
            goal[0] = scene->detour[0];
            goal[1] = scene->detour[1];
        }
    }
    command_toward(
        position,
        quaternion,
        goal,
        0.52f,
        max_yawrate_deg_s,
        action
    );
}
