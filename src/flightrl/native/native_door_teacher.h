#ifndef FLIGHTRL_NATIVE_DOOR_TEACHER_H
#define FLIGHTRL_NATIVE_DOOR_TEACHER_H

#include "native_door_scene.h"

void flightrl_door_teacher_advance(
    const float *position,
    const float *quaternion,
    FlightRLDoorScene *scene
);

void flightrl_door_teacher_action(
    const float *position,
    const float *quaternion,
    FlightRLDoorScene *scene,
    float max_yawrate_deg_s,
    float *action
);

#endif
