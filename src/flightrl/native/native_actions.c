#include "native_env.h"

static void flightrl_map_stabilized_action(DronePlanarEnv *env, const float *raw_action) {
    RuntimeDynamics *dyn = &env->runtime_dynamics;
    float thrust_cmd = flightrl_clamp(raw_action[0], -1.0f, 1.0f);
    float torque_cmd = flightrl_clamp(raw_action[1], -1.0f, 1.0f);

    float total_thrust = dyn->hover_thrust + (thrust_cmd * dyn->thrust_gain);
    total_thrust = flightrl_clamp(total_thrust, 0.0f, dyn->max_total_thrust);

    float torque = torque_cmd * dyn->max_pitch_torque;
    float left = (0.5f * total_thrust) - (0.5f * torque / dyn->arm_length);
    float right = (0.5f * total_thrust) + (0.5f * torque / dyn->arm_length);

    env->action_state[0] = left;
    env->action_state[1] = right;
}

static void flightrl_map_motor_action(DronePlanarEnv *env, const float *raw_action) {
    RuntimeDynamics *dyn = &env->runtime_dynamics;
    float left = 0.5f * (flightrl_clamp(raw_action[0], -1.0f, 1.0f) + 1.0f);
    float right = 0.5f * (flightrl_clamp(raw_action[1], -1.0f, 1.0f) + 1.0f);

    env->action_state[0] = left * 0.5f * dyn->max_total_thrust;
    env->action_state[1] = right * 0.5f * dyn->max_total_thrust;
}

void flightrl_apply_action(DronePlanarEnv *env, const float *raw_action) {
    RuntimeDynamics *dyn = &env->runtime_dynamics;
    const float alpha = dyn->actuator_tau <= 0.0f ? 1.0f : env->dt / (dyn->actuator_tau + env->dt);

    env->current_action[0] = flightrl_clamp(raw_action[0], -1.0f, 1.0f);
    env->current_action[1] = flightrl_clamp(raw_action[1], -1.0f, 1.0f);

    if (env->task_config.action_mode == FLIGHT_ACTION_STABILIZED) {
        flightrl_map_stabilized_action(env, env->current_action);
    } else {
        flightrl_map_motor_action(env, env->current_action);
    }

    for (int i = 0; i < 2; ++i) {
        env->motor_thrusts[i] += alpha * (env->action_state[i] - env->motor_thrusts[i]);
        env->motor_thrusts[i] = flightrl_clamp(env->motor_thrusts[i], 0.0f, 0.5f * dyn->max_total_thrust);
    }
}
