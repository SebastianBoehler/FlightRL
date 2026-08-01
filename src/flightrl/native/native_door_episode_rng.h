#ifndef FLIGHTRL_NATIVE_DOOR_EPISODE_RNG_H
#define FLIGHTRL_NATIVE_DOOR_EPISODE_RNG_H

#include <stdint.h>

#define FLIGHTRL_DOOR_GROUP_SCHEMA_VERSION 1u

typedef struct {
    uint32_t base_seed;
    uint32_t appearance_seed;
    uint32_t env_index;
    uint64_t next_episode_index;
} FlightRLDoorEpisodeRng;

typedef struct {
    float layout_episode_fraction[3];
    float layout_success_fraction[3];
    float door_face_episode_fraction[3];
    float door_face_success_fraction[3];
    float low_light_episode_fraction;
    float low_light_success_fraction;
    float obstacle_episode_fraction;
    float obstacle_success_fraction;
} FlightRLDoorGroupLog;

void flightrl_door_episode_rng_init(
    FlightRLDoorEpisodeRng *state,
    uint32_t base_seed,
    uint32_t appearance_seed,
    uint32_t env_index
);

uint64_t flightrl_door_episode_rng_next(
    FlightRLDoorEpisodeRng *state,
    uint32_t *physical_rng,
    uint32_t *appearance_rng
);

uint32_t flightrl_door_seed_mix(uint32_t value);

uint8_t flightrl_door_scene_group_id(
    uint8_t layout_family,
    uint8_t door_face,
    uint8_t low_light,
    uint8_t obstacle_present,
    uint8_t initial_outside_fov
);

void flightrl_door_group_log_add(
    FlightRLDoorGroupLog *log,
    uint8_t group_id,
    uint8_t success
);

#endif
