#ifndef FLIGHTRL_NATIVE_DOOR_SCENE_H
#define FLIGHTRL_NATIVE_DOOR_SCENE_H

#include <stdint.h>

typedef struct {
    float door[6];
    float obstacle[6];
    float target[3];
    float target_yaw;
    float teacher_side;
    float detour[2];
    float coverage[2];
    float search_yaw;
    float search_yaw_progress;
    unsigned char detour_active;
    unsigned char initial_outside_fov;
    unsigned char target_observed;
    unsigned char search_phase;
} FlightRLDoorScene;

void flightrl_door_scene_sample(
    float *position,
    float *quaternion,
    const float *room,
    FlightRLDoorScene *scene,
    uint32_t *rng,
    float obstacle_probability,
    float layout_diversity
);

float flightrl_door_scene_distance(
    const float *position,
    const FlightRLDoorScene *scene
);

int flightrl_door_scene_collides(
    const float *position,
    const float *room,
    const FlightRLDoorScene *scene
);

int flightrl_door_scene_visible(
    const float *position,
    const float *quaternion,
    const float *room,
    const FlightRLDoorScene *scene
);

float flightrl_door_scene_center_x(
    const float *position,
    const float *quaternion,
    const float *room,
    const FlightRLDoorScene *scene
);

#endif
