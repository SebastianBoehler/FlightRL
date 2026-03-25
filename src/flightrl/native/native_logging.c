#include "native_env.h"

void flightrl_record_step(DronePlanarEnv *env) {
    env->distance_sum += env->current_distance;
    env->action_magnitude_sum += flightrl_vector_norm(env->current_action, env->sensor_config.action_dim);
}

void flightrl_finalize_episode(DronePlanarEnv *env, int reason) {
    float steps = env->step_count > 0 ? (float)env->step_count : 1.0f;

    env->log.episode_return += env->episode_return;
    env->log.episode_length += steps;
    env->log.success_rate += reason == FLIGHT_TERM_SUCCESS ? 1.0f : 0.0f;
    env->log.crash_rate += reason == FLIGHT_TERM_CRASH ? 1.0f : 0.0f;
    env->log.timeout_rate += reason == FLIGHT_TERM_TIMEOUT ? 1.0f : 0.0f;
    env->log.mean_distance += env->distance_sum / steps;
    env->log.mean_action_magnitude += env->action_magnitude_sum / steps;
    env->log.n += 1.0f;
}
