#ifndef FLIGHTRL_NATIVE_SIXDOF_H
#define FLIGHTRL_NATIVE_SIXDOF_H

#include <stdint.h>

void flightrl_sixdof_step_batch(
    float *position,
    float *velocity,
    float *quaternion,
    float *body_rates,
    float *ranges,
    const float *actions,
    const float *room,
    int num_envs,
    float dt
);

void flightrl_sixdof_step_env_batch(
    float *position,
    float *velocity,
    float *quaternion,
    float *body_rates,
    float *ranges,
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
);

void flightrl_sixdof_step_env_context_batch(
    float *position,
    float *velocity,
    float *quaternion,
    float *body_rates,
    float *ranges,
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
    const int *task_ids,
    int reward_mode,
    const float *previous_error,
    int num_envs,
    float dt
);

uint32_t flightrl_sixdof_reset_one(
    float *position,
    float *velocity,
    float *quaternion,
    float *body_rates,
    float *ranges,
    float *target_position,
    float *target_yaw,
    float *previous_action,
    int *step_count,
    float *observation,
    float *reward,
    unsigned char *terminal,
    unsigned char *truncation,
    const float *room,
    uint32_t rng
);

#endif
