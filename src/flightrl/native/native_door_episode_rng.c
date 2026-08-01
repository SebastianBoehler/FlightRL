#include "native_door_episode_rng.h"

#define DOOR_PHYSICAL_SEED_DOMAIN 0x50485953u
#define DOOR_APPEARANCE_SEED_DOMAIN 0x41505052u
#define DOOR_SEED_FOLD_OFFSET 0x9e3779b9u

uint32_t flightrl_door_seed_mix(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

static uint32_t fold_seed(uint32_t state, uint32_t value) {
    return flightrl_door_seed_mix(
        state ^ flightrl_door_seed_mix(value + DOOR_SEED_FOLD_OFFSET)
    );
}

static uint32_t physical_episode_seed(
    const FlightRLDoorEpisodeRng *state,
    uint64_t episode_index
) {
    uint32_t seed = flightrl_door_seed_mix(DOOR_PHYSICAL_SEED_DOMAIN);
    seed = fold_seed(seed, state->base_seed);
    seed = fold_seed(seed, state->env_index);
    seed = fold_seed(seed, (uint32_t)episode_index);
    return fold_seed(seed, (uint32_t)(episode_index >> 32));
}

static uint32_t appearance_episode_seed(
    const FlightRLDoorEpisodeRng *state,
    uint64_t episode_index
) {
    uint32_t seed = flightrl_door_seed_mix(DOOR_APPEARANCE_SEED_DOMAIN);
    seed = fold_seed(seed, state->base_seed);
    seed = fold_seed(seed, state->appearance_seed);
    seed = fold_seed(seed, state->env_index);
    seed = fold_seed(seed, (uint32_t)episode_index);
    return fold_seed(seed, (uint32_t)(episode_index >> 32));
}

void flightrl_door_episode_rng_init(
    FlightRLDoorEpisodeRng *state,
    uint32_t base_seed,
    uint32_t appearance_seed,
    uint32_t env_index
) {
    state->base_seed = base_seed;
    state->appearance_seed = appearance_seed;
    state->env_index = env_index;
    state->next_episode_index = 0;
}

uint64_t flightrl_door_episode_rng_next(
    FlightRLDoorEpisodeRng *state,
    uint32_t *physical_rng,
    uint32_t *appearance_rng
) {
    uint64_t episode_index = state->next_episode_index++;
    *physical_rng = physical_episode_seed(state, episode_index);
    *appearance_rng = appearance_episode_seed(state, episode_index);
    return episode_index;
}

uint8_t flightrl_door_scene_group_id(
    uint8_t layout_family,
    uint8_t door_face,
    uint8_t low_light,
    uint8_t obstacle_present,
    uint8_t initial_outside_fov
) {
    return (uint8_t)(
        (layout_family & 3u)
        | ((door_face & 3u) << 2)
        | ((low_light != 0u) << 4)
        | ((obstacle_present != 0u) << 5)
        | ((initial_outside_fov != 0u) << 6)
    );
}

void flightrl_door_group_log_add(
    FlightRLDoorGroupLog *log,
    uint8_t group_id,
    uint8_t success
) {
    uint8_t layout_family = group_id & 3u;
    uint8_t door_face = (group_id >> 2) & 3u;
    float passed = success != 0u ? 1.0f : 0.0f;
    if (layout_family > 0u) {
        log->layout_episode_fraction[layout_family - 1u] += 1.0f;
        log->layout_success_fraction[layout_family - 1u] += passed;
    }
    if (door_face > 0u) {
        log->door_face_episode_fraction[door_face - 1u] += 1.0f;
        log->door_face_success_fraction[door_face - 1u] += passed;
    }
    if ((group_id & (1u << 4)) != 0u) {
        log->low_light_episode_fraction += 1.0f;
        log->low_light_success_fraction += passed;
    }
    if ((group_id & (1u << 5)) != 0u) {
        log->obstacle_episode_fraction += 1.0f;
        log->obstacle_success_fraction += passed;
    }
}
