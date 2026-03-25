#include "native_env.h"

float flightrl_task_distance(DronePlanarEnv *env) {
    TargetPoint target = env->world.targets[env->world.active_target];
    return flightrl_norm2(env->drone.x - target.x, env->drone.z - target.z);
}

int flightrl_task_step(DronePlanarEnv *env) {
    float speed = flightrl_norm2(env->drone.vx, env->drone.vz);
    env->current_distance = flightrl_task_distance(env);

    if (env->task_config.task_type == FLIGHT_TASK_HOVER) {
        if (env->current_distance < env->task_config.success_radius && speed < env->task_config.hover_speed_threshold) {
            env->world.hold_steps += 1;
        } else {
            env->world.hold_steps = 0;
        }
        return env->world.hold_steps >= env->task_config.hover_hold_steps ? 2 : 0;
    }

    if (env->current_distance >= env->task_config.success_radius) {
        return 0;
    }

    if (env->task_config.task_type == FLIGHT_TASK_REACH) {
        return 2;
    }

    env->world.active_target += 1;
    return env->world.active_target >= env->world.target_count ? 2 : 1;
}
