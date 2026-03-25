#include "native_env.h"

void flightrl_update_wind(DronePlanarEnv *env) {
    WindConfig *cfg = &env->wind_config;
    if (!cfg->enabled) {
        env->gust_x = 0.0f;
        env->gust_z = 0.0f;
        env->wind_x = 0.0f;
        env->wind_z = 0.0f;
        return;
    }

    if (cfg->gust_strength <= 0.0f) {
        env->gust_x = 0.0f;
        env->gust_z = 0.0f;
    } else if (cfg->gust_tau <= 0.0f) {
        env->gust_x = flightrl_rng_symmetric(&env->rng_state, cfg->gust_strength);
        env->gust_z = flightrl_rng_symmetric(&env->rng_state, cfg->gust_strength);
    } else {
        float alpha = env->dt / (cfg->gust_tau + env->dt);
        float target_x = flightrl_rng_symmetric(&env->rng_state, cfg->gust_strength);
        float target_z = flightrl_rng_symmetric(&env->rng_state, cfg->gust_strength);
        env->gust_x += alpha * (target_x - env->gust_x);
        env->gust_z += alpha * (target_z - env->gust_z);
    }

    env->wind_x = cfg->steady_x + env->gust_x;
    env->wind_z = cfg->steady_z + env->gust_z;
}
