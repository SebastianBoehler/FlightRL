#include "native_env.h"

void flightrl_update_reward(DronePlanarEnv *env, int event_code) {
    RewardConfig *cfg = &env->reward_config;
    float speed = flightrl_norm2(env->drone.vx, env->drone.vz);
    float delta[FLIGHTRL_MAX_ACTION_DIM] = {0};
    for (int i = 0; i < env->sensor_config.action_dim; ++i) {
        delta[i] = env->current_action[i] - env->previous_action[i];
    }
    float action_mag = flightrl_vector_norm(env->current_action, env->sensor_config.action_dim);
    float action_delta = flightrl_vector_norm(delta, env->sensor_config.action_dim);

    env->reward_breakdown = (RewardBreakdown){
        .alive = cfg->alive_bonus,
        .distance_penalty = -cfg->distance_penalty * env->current_distance,
        .progress_bonus = cfg->progress_bonus * (env->world.prev_distance - env->current_distance),
        .velocity_penalty = -cfg->velocity_penalty * speed,
        .angular_rate_penalty = -cfg->angular_rate_penalty * fabsf(env->drone.pitch_rate),
        .control_penalty = -cfg->control_penalty * action_mag,
        .smoothness_penalty = -cfg->smoothness_penalty * action_delta,
        .success_bonus = event_code > 0 ? cfg->success_bonus : 0.0f,
        .crash_penalty = env->terminal_reason == FLIGHT_TERM_CRASH ? -cfg->crash_penalty : 0.0f,
        .out_of_bounds_penalty = env->terminal_reason == FLIGHT_TERM_OUT_OF_BOUNDS ? -cfg->out_of_bounds_penalty : 0.0f,
        .total = 0.0f,
    };

    env->reward_breakdown.total =
        env->reward_breakdown.alive +
        env->reward_breakdown.distance_penalty +
        env->reward_breakdown.progress_bonus +
        env->reward_breakdown.velocity_penalty +
        env->reward_breakdown.angular_rate_penalty +
        env->reward_breakdown.control_penalty +
        env->reward_breakdown.smoothness_penalty +
        env->reward_breakdown.success_bonus +
        env->reward_breakdown.crash_penalty +
        env->reward_breakdown.out_of_bounds_penalty;

    env->rewards[0] = env->reward_breakdown.total;
    env->episode_return += env->reward_breakdown.total;
}
