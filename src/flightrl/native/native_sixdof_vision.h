#ifndef FLIGHTRL_NATIVE_SIXDOF_VISION_H
#define FLIGHTRL_NATIVE_SIXDOF_VISION_H

#include <stdint.h>

#define SIXDOF_VISION_WIDTH 64
#define SIXDOF_VISION_HEIGHT 48
#define SIXDOF_VISION_PIXELS (SIXDOF_VISION_WIDTH * SIXDOF_VISION_HEIGHT)
#define SIXDOF_VISION_CHANNELS 3
#define SIXDOF_VISION_INTENT_DIM 6
#define SIXDOF_VISION_OBS_DIM (SIXDOF_VISION_PIXELS * SIXDOF_VISION_CHANNELS + SIXDOF_VISION_INTENT_DIM)

void flightrl_sixdof_render_gray4_batch(
    const float *position,
    const float *quaternion,
    const float *room,
    const float *target_mean,
    const int *scene_seed,
    uint8_t *frames,
    int num_envs
);

void flightrl_sixdof_visual_observation_batch(
    const float *position,
    const float *quaternion,
    const float *target_position,
    const float *target_yaw,
    const float *room,
    const float *target_mean,
    const int *scene_seed,
    uint8_t *previous_frame,
    const uint8_t *reset_temporal,
    float *observations,
    int num_envs
);

void flightrl_sixdof_visual_observation_scene(
    const float *position,
    const float *quaternion,
    const float *target_position,
    float target_yaw,
    const float *room,
    const float *obstacle,
    float target_mean,
    int scene_seed,
    uint8_t *previous_frame,
    uint8_t reset_temporal,
    float *observation
);

#endif
