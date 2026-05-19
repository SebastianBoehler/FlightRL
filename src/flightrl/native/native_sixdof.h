#ifndef FLIGHTRL_NATIVE_SIXDOF_H
#define FLIGHTRL_NATIVE_SIXDOF_H

void flightrl_sixdof_step_batch(
    float *position,
    float *velocity,
    float *quaternion,
    float *body_rates,
    float *ranges,
    const float *actions,
    int num_envs,
    float dt
);

#endif
