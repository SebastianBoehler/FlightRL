#include <math.h>
#include <string.h>

#include "native_edge_student_observation.h"

static float edge_observation_clamp(float value, float low, float high) {
    return value < low ? low : (value > high ? high : value);
}

static void edge_quaternion_matrix(const float *q, float r[9]) {
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

static float edge_yaw(const float *q) {
    return atan2f(
        2.0f * (q[0] * q[3] + q[1] * q[2]),
        1.0f - 2.0f * (q[2] * q[2] + q[3] * q[3])
    );
}

void flightrl_edge_student_telemetry(
    const float *position,
    const float *velocity,
    const float *quaternion,
    const float *body_rates,
    float takeoff_origin_z,
    const float *mission_origin_position,
    float mission_origin_yaw,
    const float *previous_applied_action,
    float *telemetry
) {
    float rotation[9];
    edge_quaternion_matrix(quaternion, rotation);
    telemetry[0] = edge_observation_clamp(
        rotation[0] * velocity[0] + rotation[3] * velocity[1]
            + rotation[6] * velocity[2],
        -1.0f, 1.0f
    );
    telemetry[1] = edge_observation_clamp(
        rotation[1] * velocity[0] + rotation[4] * velocity[1]
            + rotation[7] * velocity[2],
        -1.0f, 1.0f
    );
    telemetry[2] = edge_observation_clamp(
        2.0f * (rotation[2] * velocity[0] + rotation[5] * velocity[1]
            + rotation[8] * velocity[2]),
        -1.0f, 1.0f
    );
    telemetry[3] = edge_observation_clamp(body_rates[0] / 6.0f, -1.0f, 1.0f);
    telemetry[4] = edge_observation_clamp(body_rates[1] / 6.0f, -1.0f, 1.0f);
    telemetry[5] = edge_observation_clamp(body_rates[2] / 4.0f, -1.0f, 1.0f);
    telemetry[6] = rotation[6];
    telemetry[7] = rotation[7];
    telemetry[8] = rotation[8];
    telemetry[9] = edge_observation_clamp(
        (position[2] - takeoff_origin_z) / 2.5f,
        0.0f, 1.0f
    );
    float dx = position[0] - mission_origin_position[0];
    float dy = position[1] - mission_origin_position[1];
    float cosine = cosf(mission_origin_yaw);
    float sine = sinf(mission_origin_yaw);
    telemetry[10] = edge_observation_clamp(
        (cosine * dx + sine * dy) / 4.0f, -1.0f, 1.0f
    );
    telemetry[11] = edge_observation_clamp(
        (-sine * dx + cosine * dy) / 4.0f, -1.0f, 1.0f
    );
    telemetry[12] = edge_observation_clamp(
        (position[2] - mission_origin_position[2]) / 2.0f,
        -1.0f, 1.0f
    );
    float relative_yaw = edge_yaw(quaternion) - mission_origin_yaw;
    telemetry[13] = sinf(relative_yaw);
    telemetry[14] = cosf(relative_yaw);
    for (int index = 0; index < FLIGHTRL_EDGE_ACTION_DIM; ++index) {
        telemetry[15 + index] = edge_observation_clamp(
            previous_applied_action[index], -1.0f, 1.0f
        );
    }
}

void flightrl_edge_student_observation(
    const float *position,
    const float *velocity,
    const float *quaternion,
    const float *body_rates,
    const float *room,
    const float *door,
    const float *obstacle,
    float target_mean,
    int scene_seed,
    float camera_randomization,
    int control_step,
    float camera_mask,
    float takeoff_origin_z,
    const float *mission_origin_position,
    float mission_origin_yaw,
    const float *previous_applied_action,
    float *grounding,
    float *observation
) {
    flightrl_sixdof_edge_door_observation_scene(
        position, quaternion, room, door, obstacle, target_mean, scene_seed,
        camera_randomization, control_step, grounding, observation
    );
    if (camera_mask > 0.5f) {
        memset(observation, 0, sizeof(float) * SIXDOF_VISION_PIXELS);
        memset(grounding, 0, sizeof(float) * FLIGHTRL_EDGE_GROUNDING_DIM);
    }
    flightrl_edge_student_telemetry(
        position, velocity, quaternion, body_rates, takeoff_origin_z,
        mission_origin_position, mission_origin_yaw, previous_applied_action,
        observation + SIXDOF_VISION_PIXELS
    );
    int target_offset = SIXDOF_VISION_PIXELS + FLIGHTRL_EDGE_TELEMETRY_DIM;
    observation[target_offset] = 1.0f;
    observation[target_offset + 1] = 0.0f;
    observation[target_offset + 2] = 0.0f;
}

int flightrl_edge_student_update_target_observed(
    unsigned char reset_temporal,
    float rendered_visible,
    unsigned char *target_observed,
    unsigned char *initial_outside_fov
) {
    int visible = rendered_visible > 0.5f;
    if (reset_temporal) {
        *target_observed = (unsigned char)visible;
        *initial_outside_fov = (unsigned char)!visible;
    } else if (visible) {
        *target_observed = 1;
    }
    return visible;
}

void flightrl_edge_student_training_tail(
    const float *teacher_action,
    const float *grounding,
    uint8_t scene_group_id,
    float *observation
) {
    int tail = FLIGHTRL_EDGE_ACTOR_OBS_DIM;
    observation[tail] = teacher_action[0];
    observation[tail + 1] = 0.0f;
    observation[tail + 2] = 0.0f;
    observation[tail + 3] = teacher_action[1];
    memcpy(
        observation + tail + FLIGHTRL_EDGE_ACTION_DIM,
        grounding,
        sizeof(float) * FLIGHTRL_EDGE_GROUNDING_DIM
    );
    flightrl_edge_student_scene_group_tail(scene_group_id, observation);
}

void flightrl_edge_student_scene_group_tail(
    uint8_t scene_group_id,
    float *observation
) {
    int offset = (
        FLIGHTRL_EDGE_ACTOR_OBS_DIM + FLIGHTRL_EDGE_ACTION_DIM
        + FLIGHTRL_EDGE_GROUNDING_DIM
    );
    observation[offset] = (float)scene_group_id;
}
