#ifndef FLIGHTRL_NATIVE_SIXDOF_H
#define FLIGHTRL_NATIVE_SIXDOF_H

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

#endif
