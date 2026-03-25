#include "native_env.h"

static void flightrl_push(float *buffer, int *index, float value) {
    buffer[*index] = value;
    *index += 1;
}

static float flightrl_noisy(DronePlanarEnv *env, float value, float scale) {
    float std = scale * env->sensor_noise_multiplier;
    return value + flightrl_rng_symmetric(&env->rng_state, std);
}

void flightrl_fill_observation(DronePlanarEnv *env) {
    int idx = 0;
    float *obs = env->observations;
    int flags = env->sensor_config.flags;
    RuntimeDynamics *dyn = &env->runtime_dynamics;
    TargetPoint target = env->world.targets[env->world.active_target];
    float rel_x = (target.x - env->drone.x) / dyn->x_limit;
    float rel_z = (target.z - env->drone.z) / dyn->z_limit;

    if (flags & FLIGHT_OBS_POSITION) {
        flightrl_push(obs, &idx, env->drone.x / dyn->x_limit);
        flightrl_push(obs, &idx, env->drone.z / dyn->z_limit);
    }
    if (flags & FLIGHT_OBS_VELOCITY) {
        flightrl_push(obs, &idx, env->drone.vx / dyn->max_velocity);
        flightrl_push(obs, &idx, env->drone.vz / dyn->max_velocity);
    }
    if (flags & FLIGHT_OBS_ATTITUDE) {
        flightrl_push(obs, &idx, sinf(env->drone.pitch));
        flightrl_push(obs, &idx, cosf(env->drone.pitch));
    }
    if (flags & FLIGHT_OBS_ANGULAR_VELOCITY) {
        flightrl_push(obs, &idx, env->drone.pitch_rate / dyn->max_pitch_rate);
    }
    if (flags & FLIGHT_OBS_TARGET_VECTOR) {
        flightrl_push(obs, &idx, rel_x);
        flightrl_push(obs, &idx, rel_z);
    }
    if (flags & FLIGHT_OBS_PREVIOUS_ACTION) {
        for (int i = 0; i < env->sensor_config.action_dim; ++i) {
            flightrl_push(obs, &idx, env->previous_action[i]);
        }
    }
    if (flags & FLIGHT_OBS_HEALTH) {
        flightrl_push(obs, &idx, 1.0f);
    }
    if (flags & FLIGHT_OBS_IDEAL_STATE) {
        flightrl_push(obs, &idx, env->drone.x / dyn->x_limit);
        flightrl_push(obs, &idx, env->drone.z / dyn->z_limit);
        flightrl_push(obs, &idx, env->drone.vx / dyn->max_velocity);
        flightrl_push(obs, &idx, env->drone.vz / dyn->max_velocity);
        flightrl_push(obs, &idx, sinf(env->drone.pitch));
        flightrl_push(obs, &idx, env->drone.pitch_rate / dyn->max_pitch_rate);
    }
    if (flags & FLIGHT_OBS_NOISY_STATE) {
        float scale = env->sensor_config.state_noise_std;
        flightrl_push(obs, &idx, flightrl_noisy(env, env->drone.x / dyn->x_limit, scale));
        flightrl_push(obs, &idx, flightrl_noisy(env, env->drone.z / dyn->z_limit, scale));
        flightrl_push(obs, &idx, flightrl_noisy(env, env->drone.vx / dyn->max_velocity, scale));
        flightrl_push(obs, &idx, flightrl_noisy(env, env->drone.vz / dyn->max_velocity, scale));
        flightrl_push(obs, &idx, flightrl_noisy(env, sinf(env->drone.pitch), scale));
        flightrl_push(obs, &idx, flightrl_noisy(env, env->drone.pitch_rate / dyn->max_pitch_rate, scale));
    }
    if (flags & FLIGHT_OBS_IMU) {
        float scale = env->sensor_config.imu_noise_std;
        flightrl_push(obs, &idx, flightrl_noisy(env, env->last_ax / dyn->gravity, scale));
        flightrl_push(obs, &idx, flightrl_noisy(env, env->last_az / dyn->gravity, scale));
        flightrl_push(obs, &idx, flightrl_noisy(env, env->drone.pitch_rate / dyn->max_pitch_rate, scale));
    }

    while (idx < env->sensor_config.observation_dim) {
        obs[idx++] = 0.0f;
    }
}
