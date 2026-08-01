#ifndef FLIGHTRL_NATIVE_EDGE_STUDENT_OBSERVATION_H
#define FLIGHTRL_NATIVE_EDGE_STUDENT_OBSERVATION_H

#include "native_sixdof_vision.h"

#if SIXDOF_VISION_WIDTH != 64 || SIXDOF_VISION_HEIGHT != 48
#error "edge-v3 student requires an exact 64x48 frame"
#endif

#define FLIGHTRL_EDGE_TELEMETRY_DIM 19
#define FLIGHTRL_EDGE_TARGET_DIM 3
#define FLIGHTRL_EDGE_ACTION_DIM 4
#define FLIGHTRL_EDGE_GROUNDING_DIM 4
#define FLIGHTRL_EDGE_ACTOR_OBS_DIM \
    (SIXDOF_VISION_PIXELS + FLIGHTRL_EDGE_TELEMETRY_DIM \
        + FLIGHTRL_EDGE_TARGET_DIM)
#define FLIGHTRL_EDGE_STUDENT_OBS_DIM \
    (FLIGHTRL_EDGE_ACTOR_OBS_DIM + FLIGHTRL_EDGE_ACTION_DIM \
        + FLIGHTRL_EDGE_GROUNDING_DIM)

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
);

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
    float camera_mask,
    float takeoff_origin_z,
    const float *mission_origin_position,
    float mission_origin_yaw,
    const float *previous_applied_action,
    float *grounding,
    float *observation
);

int flightrl_edge_student_update_target_observed(
    unsigned char reset_temporal,
    float rendered_visible,
    unsigned char *target_observed,
    unsigned char *initial_outside_fov
);

void flightrl_edge_student_training_tail(
    const float *teacher_action,
    const float *grounding,
    float *observation
);

#endif
