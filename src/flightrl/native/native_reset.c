#include "native_env.h"

static void flightrl_copy_waypoints(DronePlanarEnv *env) {
    env->world.target_count = env->task_config.sequence_length;
    for (int i = 0; i < env->world.target_count; ++i) {
        env->world.targets[i] = env->task_config.fixed_waypoints[i];
    }
}

void flightrl_sample_runtime(DronePlanarEnv *env) {
    env->runtime_dynamics = (RuntimeDynamics){
        .mass = env->drone_config.mass,
        .inertia = env->drone_config.inertia,
        .arm_length = env->drone_config.arm_length,
        .drag = env->drone_config.drag,
        .angular_drag = env->drone_config.angular_drag,
        .gravity = env->drone_config.gravity,
        .hover_thrust = env->drone_config.hover_thrust,
        .thrust_gain = env->drone_config.thrust_gain,
        .max_total_thrust = env->drone_config.max_total_thrust,
        .max_pitch_torque = env->drone_config.max_pitch_torque,
        .actuator_tau = env->drone_config.actuator_tau,
        .max_velocity = env->drone_config.max_velocity,
        .max_pitch_rate = env->drone_config.max_pitch_rate,
        .max_pitch_angle = env->drone_config.max_pitch_angle,
        .floor_z = env->drone_config.floor_z,
        .x_limit = env->drone_config.x_limit,
        .z_limit = env->drone_config.z_limit,
    };

    if (!env->randomization_config.enabled) {
        env->sensor_noise_multiplier = 1.0f;
        return;
    }

    env->runtime_dynamics.mass *= 1.0f + flightrl_rng_symmetric(&env->rng_state, env->randomization_config.mass_scale);
    env->runtime_dynamics.drag *= 1.0f + flightrl_rng_symmetric(&env->rng_state, env->randomization_config.drag_scale);
    env->runtime_dynamics.hover_thrust *= 1.0f + flightrl_rng_symmetric(&env->rng_state, env->randomization_config.thrust_scale);
    env->runtime_dynamics.thrust_gain *= 1.0f + flightrl_rng_symmetric(&env->rng_state, env->randomization_config.thrust_scale);
    env->runtime_dynamics.actuator_tau *= 1.0f + flightrl_rng_symmetric(&env->rng_state, env->randomization_config.actuator_tau_scale);
    env->sensor_noise_multiplier = 1.0f + flightrl_rng_symmetric(&env->rng_state, env->randomization_config.sensor_noise_scale);
}

void flightrl_reset_episode_stats(DronePlanarEnv *env) {
    env->step_count = 0;
    env->terminal_reason = FLIGHT_TERM_NONE;
    env->episode_return = 0.0f;
    env->distance_sum = 0.0f;
    env->action_magnitude_sum = 0.0f;
    env->reward_breakdown = (RewardBreakdown){0};
}

void flightrl_reset_state(DronePlanarEnv *env) {
    TaskConfig *task = &env->task_config;
    int deterministic = task->reset_mode == FLIGHT_RESET_DETERMINISTIC;

    flightrl_sample_runtime(env);
    flightrl_reset_episode_stats(env);

    env->drone = (DroneState){
        .x = deterministic ? task->fixed_start_x : flightrl_rng_uniform(&env->rng_state, task->spawn_x_min, task->spawn_x_max),
        .z = deterministic ? task->fixed_start_z : flightrl_rng_uniform(&env->rng_state, task->spawn_z_min, task->spawn_z_max),
        .vx = 0.0f,
        .vz = 0.0f,
        .pitch = 0.0f,
        .pitch_rate = 0.0f,
    };

    env->motor_thrusts[0] = 0.5f * env->runtime_dynamics.hover_thrust;
    env->motor_thrusts[1] = 0.5f * env->runtime_dynamics.hover_thrust;
    env->previous_action[0] = 0.0f;
    env->previous_action[1] = 0.0f;
    env->current_action[0] = 0.0f;
    env->current_action[1] = 0.0f;
    env->last_ax = 0.0f;
    env->last_az = 0.0f;

    if (task->task_type == FLIGHT_TASK_SEQUENCE && deterministic) {
        flightrl_copy_waypoints(env);
    } else {
        env->world.target_count = task->task_type == FLIGHT_TASK_SEQUENCE ? task->sequence_length : 1;
        for (int i = 0; i < env->world.target_count; ++i) {
            env->world.targets[i].x = deterministic ? task->fixed_target_x : flightrl_rng_uniform(&env->rng_state, task->target_x_min, task->target_x_max);
            env->world.targets[i].z = deterministic ? task->fixed_target_z : flightrl_rng_uniform(&env->rng_state, task->target_z_min, task->target_z_max);
        }
    }

    env->world.active_target = 0;
    env->world.hold_steps = 0;
    env->world.prev_distance = flightrl_task_distance(env);
    env->current_distance = env->world.prev_distance;
}
