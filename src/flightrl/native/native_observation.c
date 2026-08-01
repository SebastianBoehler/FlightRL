#include "native_env.h"

static void flightrl_push(float *buffer, int *index, float value) {
    buffer[*index] = value;
    *index += 1;
}

static float flightrl_noisy(DronePlanarEnv *env, float value, float scale) {
    float std = scale * env->sensor_noise_multiplier;
    return value + flightrl_rng_symmetric(&env->rng_state, std);
}

static float flightrl_norm_range(float value, float max_range) {
    return flightrl_clamp(value / max_range, 0.0f, 1.0f);
}

static void flightrl_range_values(DronePlanarEnv *env, float ranges[6]) {
    RuntimeDynamics *dyn = &env->runtime_dynamics;
    ranges[0] = dyn->x_limit - env->drone.x;
    ranges[1] = env->drone.x + dyn->x_limit;
    ranges[2] = 4.0f;
    ranges[3] = 4.0f;
    ranges[4] = dyn->z_limit - env->drone.z;
    ranges[5] = env->drone.z - dyn->floor_z;
}

static void flightrl_range_rates(DronePlanarEnv *env, float rates[6]) {
    rates[0] = -env->drone.vx;
    rates[1] = env->drone.vx;
    rates[2] = 0.0f;
    rates[3] = 0.0f;
    rates[4] = -env->drone.vz;
    rates[5] = env->drone.vz;
}

static void flightrl_push_range(DronePlanarEnv *env, int *idx) {
    float *obs = env->observations;
    float ranges[6];
    flightrl_range_values(env, ranges);
    for (int i = 0; i < 5; ++i) {
        flightrl_push(obs, idx, flightrl_norm_range(ranges[i], 4.0f));
    }
}

static float flightrl_ttc(float distance, float rate) {
    float closing_speed = fmaxf(-rate, 0.0f);
    if (closing_speed <= 1e-6f) {
        return 1e30f;
    }
    return fmaxf(distance, 0.0f) / closing_speed;
}

static void flightrl_push_range_rate(DronePlanarEnv *env, int *idx) {
    float rates[6];
    flightrl_range_rates(env, rates);
    for (int i = 0; i < 6; ++i) {
        flightrl_push(env->observations, idx, flightrl_clamp(rates[i] / 4.0f, -1.0f, 1.0f));
    }
}

static void flightrl_push_ttc(DronePlanarEnv *env, int *idx) {
    const float horizon_s = 0.7f;
    float ranges[6];
    float rates[6];
    float min_range;
    float min_ttc;
    flightrl_range_values(env, ranges);
    flightrl_range_rates(env, rates);
    min_range = ranges[0];
    min_ttc = flightrl_ttc(ranges[0], rates[0]);
    for (int i = 1; i < 4; ++i) {
        min_range = fminf(min_range, ranges[i]);
        min_ttc = fminf(min_ttc, flightrl_ttc(ranges[i], rates[i]));
    }
    flightrl_push(env->observations, idx, flightrl_norm_range(min_range, 4.0f));
    flightrl_push(
        env->observations,
        idx,
        min_ttc < horizon_s
            ? flightrl_clamp((horizon_s - fmaxf(min_ttc, 0.0f)) / horizon_s, 0.0f, 1.0f)
            : 0.0f
    );
}

static void flightrl_push_crazyflie_telemetry(DronePlanarEnv *env, int *idx, float rel_x, float rel_z) {
    float *obs = env->observations;
    RuntimeDynamics *dyn = &env->runtime_dynamics;
    float scale = env->sensor_config.imu_noise_std;

    flightrl_push(obs, idx, env->drone.x / dyn->x_limit);
    flightrl_push(obs, idx, 0.0f);
    flightrl_push(obs, idx, env->drone.z / dyn->z_limit);
    flightrl_push(obs, idx, env->drone.vx / dyn->max_velocity);
    flightrl_push(obs, idx, 0.0f);
    flightrl_push(obs, idx, env->drone.vz / dyn->max_velocity);
    flightrl_push(obs, idx, 0.0f);
    flightrl_push(obs, idx, 1.0f);
    flightrl_push(obs, idx, sinf(env->drone.pitch));
    flightrl_push(obs, idx, cosf(env->drone.pitch));
    flightrl_push(obs, idx, 0.0f);
    flightrl_push(obs, idx, 1.0f);
    flightrl_push(obs, idx, 0.0f);
    flightrl_push(obs, idx, env->drone.pitch_rate / dyn->max_pitch_rate);
    flightrl_push(obs, idx, 0.0f);
    flightrl_push(obs, idx, flightrl_noisy(env, env->last_ax / dyn->gravity, scale));
    flightrl_push(obs, idx, 0.0f);
    flightrl_push(obs, idx, flightrl_noisy(env, env->last_az / dyn->gravity, scale));
    flightrl_push_range(env, idx);
    flightrl_push(obs, idx, flightrl_norm_range(env->drone.z - dyn->floor_z, 4.0f));
    flightrl_push(obs, idx, 1.0f);
    flightrl_push(obs, idx, rel_x);
    flightrl_push(obs, idx, 0.0f);
    flightrl_push(obs, idx, rel_z);
    for (int i = 0; i < env->sensor_config.action_dim; ++i) {
        flightrl_push(obs, idx, env->previous_action[i]);
    }
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
    if (flags & FLIGHT_OBS_RANGE) {
        flightrl_push_range(env, &idx);
    }
    if (flags & FLIGHT_OBS_RANGE_RATE) {
        flightrl_push_range_rate(env, &idx);
    }
    if (flags & FLIGHT_OBS_TTC) {
        flightrl_push_ttc(env, &idx);
    }
    if (flags & FLIGHT_OBS_CRAZYFLIE_TELEMETRY) {
        flightrl_push_crazyflie_telemetry(env, &idx, rel_x, rel_z);
    }

    while (idx < env->sensor_config.observation_dim) {
        obs[idx++] = 0.0f;
    }
}
