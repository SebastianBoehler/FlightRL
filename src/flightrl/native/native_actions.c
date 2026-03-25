#include "native_env.h"

static void flightrl_set_pair_targets(DronePlanarEnv *env, float front_pair, float rear_pair) {
    env->action_state[FLIGHT_ROTOR_FRONT_LEFT] = 0.5f * front_pair;
    env->action_state[FLIGHT_ROTOR_FRONT_RIGHT] = 0.5f * front_pair;
    env->action_state[FLIGHT_ROTOR_REAR_LEFT] = 0.5f * rear_pair;
    env->action_state[FLIGHT_ROTOR_REAR_RIGHT] = 0.5f * rear_pair;
}

static void flightrl_map_stabilized_action(DronePlanarEnv *env, const float *raw_action) {
    RuntimeDynamics *dyn = &env->runtime_dynamics;
    float thrust_cmd = flightrl_clamp(raw_action[0], -1.0f, 1.0f);
    float torque_cmd = flightrl_clamp(raw_action[1], -1.0f, 1.0f);
    float total_thrust = dyn->hover_thrust + (thrust_cmd * dyn->thrust_gain);
    float torque = torque_cmd * dyn->max_pitch_torque;
    float front_pair;
    float rear_pair;

    total_thrust = flightrl_clamp(total_thrust, 0.0f, dyn->max_total_thrust);
    front_pair = flightrl_clamp(0.5f * total_thrust - 0.5f * torque / dyn->arm_length, 0.0f, 0.5f * dyn->max_total_thrust);
    rear_pair = flightrl_clamp(0.5f * total_thrust + 0.5f * torque / dyn->arm_length, 0.0f, 0.5f * dyn->max_total_thrust);
    flightrl_set_pair_targets(env, front_pair, rear_pair);
}

static void flightrl_map_pair_action(DronePlanarEnv *env, const float *raw_action) {
    RuntimeDynamics *dyn = &env->runtime_dynamics;
    float front_pair = 0.5f * (flightrl_clamp(raw_action[0], -1.0f, 1.0f) + 1.0f);
    float rear_pair = 0.5f * (flightrl_clamp(raw_action[1], -1.0f, 1.0f) + 1.0f);
    flightrl_set_pair_targets(env, front_pair * 0.5f * dyn->max_total_thrust, rear_pair * 0.5f * dyn->max_total_thrust);
}

static void flightrl_map_quad_action(DronePlanarEnv *env, const float *raw_action) {
    RuntimeDynamics *dyn = &env->runtime_dynamics;
    float max_rotor = 0.25f * dyn->max_total_thrust;
    for (int i = 0; i < FLIGHTRL_NUM_ROTORS; ++i) {
        float normalized = 0.5f * (flightrl_clamp(raw_action[i], -1.0f, 1.0f) + 1.0f);
        env->action_state[i] = normalized * max_rotor;
    }
}

void flightrl_apply_action(DronePlanarEnv *env, const float *raw_action) {
    RuntimeDynamics *dyn = &env->runtime_dynamics;
    float max_rotor = 0.25f * dyn->max_total_thrust;
    float alpha = dyn->actuator_tau <= 0.0f ? 1.0f : env->dt / (dyn->actuator_tau + env->dt);
    int action_dim = env->sensor_config.action_dim;

    for (int i = 0; i < FLIGHTRL_MAX_ACTION_DIM; ++i) {
        env->current_action[i] = i < action_dim ? flightrl_clamp(raw_action[i], -1.0f, 1.0f) : 0.0f;
    }

    if (env->task_config.action_mode == FLIGHT_ACTION_STABILIZED) {
        flightrl_map_stabilized_action(env, env->current_action);
    } else if (env->task_config.action_mode == FLIGHT_ACTION_MOTOR_PAIR) {
        flightrl_map_pair_action(env, env->current_action);
    } else {
        flightrl_map_quad_action(env, env->current_action);
    }

    for (int i = 0; i < FLIGHTRL_NUM_ROTORS; ++i) {
        env->motor_thrusts[i] += alpha * (env->action_state[i] - env->motor_thrusts[i]);
        env->motor_thrusts[i] = flightrl_clamp(env->motor_thrusts[i], 0.0f, max_rotor);
    }
}
