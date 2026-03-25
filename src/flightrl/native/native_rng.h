#ifndef FLIGHTRL_NATIVE_RNG_H
#define FLIGHTRL_NATIVE_RNG_H

#include "native_types.h"

static inline uint64_t flightrl_rng_next(uint64_t *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return *state * 2685821657736338717ULL;
}

static inline float flightrl_rng_uniform(uint64_t *state, float low, float high) {
    const double unit = (double)(flightrl_rng_next(state) >> 11) * (1.0 / 9007199254740992.0);
    return low + (float)unit * (high - low);
}

static inline float flightrl_rng_symmetric(uint64_t *state, float scale) {
    return flightrl_rng_uniform(state, -scale, scale);
}

#endif
