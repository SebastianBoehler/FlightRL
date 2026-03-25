#include "native_env.h"

void flightrl_update_reward(DronePlanarEnv *env, int event_code) {
    RewardConfig *cfg = &env->reward_config;
    float speed = flightrl_norm2(env->drone.vx, env->drone.vz);
    float action_mag = flightrl_norm2(env->current_action[0], env->current_action[1]);
    float action_delta = flightrl_norm2(
        env->current_action[0] - env->previous_action[0],
        env->current_action[1] - env->previous_action[1]
    );

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
