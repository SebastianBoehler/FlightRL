#include <stdint.h>

static float clampf(float value, float low, float high) {
    return value < low ? low : (value > high ? high : value);
}

static uint32_t rng_next(uint32_t *rng) {
    *rng = 1664525u * (*rng) + 1013904223u;
    return *rng;
}

static float rnd(uint32_t *rng, float low, float high) {
    float unit = (float)(rng_next(rng) >> 8) / 16777215.0f;
    return low + unit * (high - low);
}

#include "../src/flightrl/native/native_door_scene.c"

float flightrl_test_door_collision_margin_m(void) {
    return FLIGHTRL_DOOR_COLLISION_MARGIN_M;
}

float flightrl_test_door_route_clearance_m(void) {
    return FLIGHTRL_DOOR_ROUTE_CLEARANCE_M;
}

void flightrl_test_door_scene_sample(
    uint32_t seed,
    float *position,
    float *quaternion,
    FlightRLDoorScene *scene
) {
    const float room[6] = {-2.0f, 2.0f, -2.0f, 2.0f, 0.0f, 2.5f};
    flightrl_door_scene_sample(
        position,
        quaternion,
        room,
        scene,
        &seed,
        1.0f,
        1.0f,
        0.8f,
        0.08f
    );
}

int flightrl_test_door_scene_route_is_clear(
    const float *position,
    const FlightRLDoorScene *scene
) {
    const float margin = FLIGHTRL_DOOR_ROUTE_CLEARANCE_M;
    float detour[3] = {
        scene->detour[0],
        scene->detour[1],
        position[2],
    };
    return (
        !point_in_obstacle(position, scene, margin)
        && !point_in_obstacle(scene->target, scene, margin)
        && !point_in_obstacle(detour, scene, margin)
        && !segment_hits_obstacle(position, detour, scene, margin)
        && !segment_hits_obstacle(detour, scene->target, scene, margin)
    );
}

int flightrl_test_door_scene_turn_is_clear(
    const float *position,
    const FlightRLDoorScene *scene
) {
    const float margin = FLIGHTRL_DOOR_ROUTE_CLEARANCE_M;
    float dx = scene->detour[0] - position[0];
    float dy = scene->detour[1] - position[1];
    float inverse_distance = 1.0f / fmaxf(hypotf(dx, dy), 1.0e-6f);
    float release[3] = {
        scene->detour[0]
            - FLIGHTRL_DOOR_DETOUR_RELEASE_RADIUS_M * dx * inverse_distance,
        scene->detour[1]
            - FLIGHTRL_DOOR_DETOUR_RELEASE_RADIUS_M * dy * inverse_distance,
        position[2],
    };
    return !segment_hits_obstacle(release, scene->target, scene, margin);
}

int flightrl_test_door_scene_coverage_is_visible(
    const FlightRLDoorScene *scene
) {
    const float room[6] = {-2.0f, 2.0f, -2.0f, 2.0f, 0.0f, 2.5f};
    const float margin = 0.0f;
    float coverage[3] = {
        scene->coverage[0],
        scene->coverage[1],
        scene->target[2],
    };
    float center[3];
    door_center(room, scene, center);
    return !segment_hits_obstacle(coverage, center, scene, margin);
}
