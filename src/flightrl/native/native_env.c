#include "native_env.h"

void c_reset(DronePlanarEnv *env) {
    env->terminals[0] = 0;
    env->truncations[0] = 0;
    env->rewards[0] = 0.0f;
    flightrl_reset_state(env);
    flightrl_fill_observation(env);
}

void c_step(DronePlanarEnv *env) {
    env->terminals[0] = 0;
    env->truncations[0] = 0;
    env->rewards[0] = 0.0f;
    env->step_count += 1;

    flightrl_apply_action(env, env->actions);
    flightrl_apply_dynamics(env);

    int event_code = flightrl_task_step(env);
    env->terminal_reason = flightrl_check_termination(env);
    if (env->terminal_reason == FLIGHT_TERM_NONE && event_code == 2) {
        env->terminal_reason = FLIGHT_TERM_SUCCESS;
    }

    flightrl_update_reward(env, event_code);
    flightrl_record_step(env);

    if (env->terminal_reason != FLIGHT_TERM_NONE) {
        if (env->terminal_reason == FLIGHT_TERM_TIMEOUT) {
            env->truncations[0] = 1;
        } else {
            env->terminals[0] = 1;
        }
        flightrl_finalize_episode(env, env->terminal_reason);
        flightrl_reset_state(env);
    } else if (event_code == 1) {
        env->world.prev_distance = flightrl_task_distance(env);
    } else {
        env->world.prev_distance = env->current_distance;
    }

    for (int i = 0; i < env->sensor_config.action_dim; ++i) {
        env->previous_action[i] = env->current_action[i];
    }
    flightrl_fill_observation(env);
}

void c_close(DronePlanarEnv *env) {
    (void)env;
}
