#ifndef FLIGHTRL_NATIVE_TYPES_H
#define FLIGHTRL_NATIVE_TYPES_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#define FLIGHTRL_MAX_WAYPOINTS 8

enum FlightTaskType {
    FLIGHT_TASK_HOVER = 0,
    FLIGHT_TASK_REACH = 1,
    FLIGHT_TASK_SEQUENCE = 2,
};

enum FlightActionMode {
    FLIGHT_ACTION_STABILIZED = 0,
    FLIGHT_ACTION_MOTOR_PAIR = 1,
};

enum FlightResetMode {
    FLIGHT_RESET_DETERMINISTIC = 0,
    FLIGHT_RESET_RANDOM_UNIFORM = 1,
};

enum FlightTerminationReason {
    FLIGHT_TERM_NONE = 0,
    FLIGHT_TERM_SUCCESS = 1,
    FLIGHT_TERM_CRASH = 2,
    FLIGHT_TERM_OUT_OF_BOUNDS = 3,
    FLIGHT_TERM_TIMEOUT = 4,
};

enum FlightObservationFlags {
    FLIGHT_OBS_POSITION = 1 << 0,
    FLIGHT_OBS_VELOCITY = 1 << 1,
    FLIGHT_OBS_ATTITUDE = 1 << 2,
    FLIGHT_OBS_ANGULAR_VELOCITY = 1 << 3,
    FLIGHT_OBS_TARGET_VECTOR = 1 << 4,
    FLIGHT_OBS_PREVIOUS_ACTION = 1 << 5,
    FLIGHT_OBS_HEALTH = 1 << 6,
    FLIGHT_OBS_IDEAL_STATE = 1 << 7,
    FLIGHT_OBS_NOISY_STATE = 1 << 8,
    FLIGHT_OBS_IMU = 1 << 9,
    FLIGHT_OBS_RANGE = 1 << 10,
    FLIGHT_OBS_VISION = 1 << 11,
};

typedef struct {
    float x;
    float z;
} TargetPoint;

typedef struct {
    float x;
    float z;
    float vx;
    float vz;
    float pitch;
    float pitch_rate;
} DroneState;

typedef struct {
    TargetPoint targets[FLIGHTRL_MAX_WAYPOINTS];
    int target_count;
    int active_target;
    int hold_steps;
    float prev_distance;
} WorldState;

typedef struct {
    float mass;
    float inertia;
    float arm_length;
    float drag;
    float angular_drag;
    float gravity;
    float hover_thrust;
    float thrust_gain;
    float max_total_thrust;
    float max_pitch_torque;
    float actuator_tau;
    float max_velocity;
    float max_pitch_rate;
    float max_pitch_angle;
    float floor_z;
    float x_limit;
    float z_limit;
} RuntimeDynamics;

typedef struct {
    float mass;
    float inertia;
    float arm_length;
    float drag;
    float angular_drag;
    float gravity;
    float hover_thrust;
    float thrust_gain;
    float max_total_thrust;
    float max_pitch_torque;
    float actuator_tau;
    float max_velocity;
    float max_pitch_rate;
    float max_pitch_angle;
    float floor_z;
    float x_limit;
    float z_limit;
} DroneConfig;

typedef struct {
    int flags;
    int action_dim;
    int observation_dim;
    float state_noise_std;
    float imu_noise_std;
} SensorConfig;

typedef struct {
    int task_type;
    int action_mode;
    int reset_mode;
    int max_steps;
    int sequence_length;
    int hover_hold_steps;
    float success_radius;
    float hover_speed_threshold;
    float fixed_start_x;
    float fixed_start_z;
    float fixed_target_x;
    float fixed_target_z;
    float spawn_x_min;
    float spawn_x_max;
    float spawn_z_min;
    float spawn_z_max;
    float target_x_min;
    float target_x_max;
    float target_z_min;
    float target_z_max;
    TargetPoint fixed_waypoints[FLIGHTRL_MAX_WAYPOINTS];
} TaskConfig;

typedef struct {
    float alive_bonus;
    float distance_penalty;
    float progress_bonus;
    float velocity_penalty;
    float angular_rate_penalty;
    float control_penalty;
    float smoothness_penalty;
    float success_bonus;
    float crash_penalty;
    float out_of_bounds_penalty;
} RewardConfig;

typedef struct {
    int enabled;
    float mass_scale;
    float drag_scale;
    float thrust_scale;
    float actuator_tau_scale;
    float sensor_noise_scale;
} DomainRandomizationConfig;

typedef struct {
    float episode_return;
    float episode_length;
    float success_rate;
    float crash_rate;
    float timeout_rate;
    float mean_distance;
    float mean_action_magnitude;
    float n;
} Log;

typedef struct {
    float alive;
    float distance_penalty;
    float progress_bonus;
    float velocity_penalty;
    float angular_rate_penalty;
    float control_penalty;
    float smoothness_penalty;
    float success_bonus;
    float crash_penalty;
    float out_of_bounds_penalty;
    float total;
} RewardBreakdown;

typedef struct DronePlanarEnv {
    Log log;
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;
    unsigned char *truncations;

    DroneConfig drone_config;
    SensorConfig sensor_config;
    TaskConfig task_config;
    RewardConfig reward_config;
    DomainRandomizationConfig randomization_config;
    RuntimeDynamics runtime_dynamics;
    DroneState drone;
    WorldState world;
    RewardBreakdown reward_breakdown;

    uint64_t rng_state;
    float dt;
    float current_distance;
    float sensor_noise_multiplier;
    unsigned int step_count;
    int terminal_reason;
    float action_state[2];
    float current_action[2];
    float motor_thrusts[2];
    float previous_action[2];
    float last_ax;
    float last_az;
    float episode_return;
    float distance_sum;
    float action_magnitude_sum;
} DronePlanarEnv;

static inline float flightrl_clamp(float value, float low, float high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static inline float flightrl_sign(float value) {
    return value >= 0.0f ? 1.0f : -1.0f;
}

static inline float flightrl_norm2(float x, float z) {
    return sqrtf((x * x) + (z * z));
}

#endif
