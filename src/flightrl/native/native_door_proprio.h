#ifndef FLIGHTRL_NATIVE_DOOR_PROPRIO_H
#define FLIGHTRL_NATIVE_DOOR_PROPRIO_H

#include "native_door_detector.h"

#define SIXDOF_DOOR_SENSOR_DIM 17
#define SIXDOF_DOOR_PHASE_DIM 4
#define SIXDOF_DOOR_PROPRIO_DIM \
    (SIXDOF_DOOR_SENSOR_DIM + SIXDOF_DOOR_PHASE_DIM \
        + SIXDOF_DOOR_EVIDENCE_DIM)

float flightrl_door_yaw(const float *quaternion);

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
);

#endif
