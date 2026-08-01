#ifndef FLIGHTRL_NATIVE_PUFFER_CONTRACT_H
#define FLIGHTRL_NATIVE_PUFFER_CONTRACT_H

#include <stdint.h>

static inline uint64_t flightrl_puffer_mix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

static inline uint64_t flightrl_puffer_seed64(uint64_t base_seed, uint32_t env_index) {
    uint64_t seed_namespace = flightrl_puffer_mix64(base_seed) | 0x8000000000000000ULL;
    return seed_namespace ^ (uint64_t)env_index;
}

static inline uint32_t flightrl_puffer_seed32(uint32_t base_seed, uint32_t env_index) {
    uint32_t seed = base_seed + 0x9e3779b9u + env_index * 0x85ebca6bu;
    seed ^= seed >> 16;
    seed *= 0x7feb352du;
    seed ^= seed >> 15;
    seed *= 0x846ca68bu;
    return seed ^ (seed >> 16);
}

#endif
