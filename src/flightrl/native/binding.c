#include "binding_env.h"
#include "binding_vec.h"

static int my_init(DronePlanarEnv *env, PyObject *kwargs) {
    env->dt = (float)flightrl_unpack_number(kwargs, "dt");
    env->sensor_config.action_dim = (int)flightrl_unpack_number(kwargs, "action_dim");
    env->sensor_config.observation_dim = (int)flightrl_unpack_number(kwargs, "observation_dim");
    env->sensor_config.flags = (int)flightrl_unpack_number(kwargs, "observation_flags");
    env->sensor_config.state_noise_std = (float)flightrl_unpack_number(kwargs, "state_noise_std");
    env->sensor_config.imu_noise_std = (float)flightrl_unpack_number(kwargs, "imu_noise_std");
    env->task_config.task_type = (int)flightrl_unpack_number(kwargs, "task_type");
    env->task_config.action_mode = (int)flightrl_unpack_number(kwargs, "action_mode");
    env->task_config.reset_mode = (int)flightrl_unpack_number(kwargs, "reset_mode");
    env->task_config.max_steps = (int)flightrl_unpack_number(kwargs, "max_steps");
    env->task_config.sequence_length = (int)flightrl_unpack_number(kwargs, "sequence_length");
    env->task_config.hover_hold_steps = (int)flightrl_unpack_number(kwargs, "hover_hold_steps");
    env->task_config.success_radius = (float)flightrl_unpack_number(kwargs, "success_radius");
    env->task_config.hover_speed_threshold = (float)flightrl_unpack_number(kwargs, "hover_speed_threshold");
    env->task_config.fixed_start_x = (float)flightrl_unpack_number(kwargs, "fixed_start_x");
    env->task_config.fixed_start_z = (float)flightrl_unpack_number(kwargs, "fixed_start_z");
    env->task_config.fixed_target_x = (float)flightrl_unpack_number(kwargs, "fixed_target_x");
    env->task_config.fixed_target_z = (float)flightrl_unpack_number(kwargs, "fixed_target_z");
    env->task_config.spawn_x_min = (float)flightrl_unpack_number(kwargs, "spawn_x_min");
    env->task_config.spawn_x_max = (float)flightrl_unpack_number(kwargs, "spawn_x_max");
    env->task_config.spawn_z_min = (float)flightrl_unpack_number(kwargs, "spawn_z_min");
    env->task_config.spawn_z_max = (float)flightrl_unpack_number(kwargs, "spawn_z_max");
    env->task_config.target_x_min = (float)flightrl_unpack_number(kwargs, "target_x_min");
    env->task_config.target_x_max = (float)flightrl_unpack_number(kwargs, "target_x_max");
    env->task_config.target_z_min = (float)flightrl_unpack_number(kwargs, "target_z_min");
    env->task_config.target_z_max = (float)flightrl_unpack_number(kwargs, "target_z_max");
    env->drone_config.mass = (float)flightrl_unpack_number(kwargs, "mass");
    env->drone_config.inertia = (float)flightrl_unpack_number(kwargs, "inertia");
    env->drone_config.arm_length = (float)flightrl_unpack_number(kwargs, "arm_length");
    env->drone_config.drag = (float)flightrl_unpack_number(kwargs, "drag");
    env->drone_config.angular_drag = (float)flightrl_unpack_number(kwargs, "angular_drag");
    env->drone_config.gravity = (float)flightrl_unpack_number(kwargs, "gravity");
    env->drone_config.hover_thrust = (float)flightrl_unpack_number(kwargs, "hover_thrust");
    env->drone_config.thrust_gain = (float)flightrl_unpack_number(kwargs, "thrust_gain");
    env->drone_config.max_total_thrust = (float)flightrl_unpack_number(kwargs, "max_total_thrust");
    env->drone_config.max_pitch_torque = (float)flightrl_unpack_number(kwargs, "max_pitch_torque");
    env->drone_config.actuator_tau = (float)flightrl_unpack_number(kwargs, "actuator_tau");
    env->drone_config.max_velocity = (float)flightrl_unpack_number(kwargs, "max_velocity");
    env->drone_config.max_pitch_rate = (float)flightrl_unpack_number(kwargs, "max_pitch_rate");
    env->drone_config.max_pitch_angle = (float)flightrl_unpack_number(kwargs, "max_pitch_angle");
    env->drone_config.floor_z = (float)flightrl_unpack_number(kwargs, "floor_z");
    env->drone_config.x_limit = (float)flightrl_unpack_number(kwargs, "x_limit");
    env->drone_config.z_limit = (float)flightrl_unpack_number(kwargs, "z_limit");
    env->reward_config.alive_bonus = (float)flightrl_unpack_number(kwargs, "alive_bonus");
    env->reward_config.distance_penalty = (float)flightrl_unpack_number(kwargs, "distance_penalty");
    env->reward_config.progress_bonus = (float)flightrl_unpack_number(kwargs, "progress_bonus");
    env->reward_config.velocity_penalty = (float)flightrl_unpack_number(kwargs, "velocity_penalty");
    env->reward_config.angular_rate_penalty = (float)flightrl_unpack_number(kwargs, "angular_rate_penalty");
    env->reward_config.control_penalty = (float)flightrl_unpack_number(kwargs, "control_penalty");
    env->reward_config.smoothness_penalty = (float)flightrl_unpack_number(kwargs, "smoothness_penalty");
    env->reward_config.success_bonus = (float)flightrl_unpack_number(kwargs, "success_bonus");
    env->reward_config.crash_penalty = (float)flightrl_unpack_number(kwargs, "crash_penalty");
    env->reward_config.out_of_bounds_penalty = (float)flightrl_unpack_number(kwargs, "out_of_bounds_penalty");
    env->randomization_config.enabled = (int)flightrl_unpack_number(kwargs, "randomization_enabled");
    env->randomization_config.mass_scale = (float)flightrl_unpack_number(kwargs, "mass_scale");
    env->randomization_config.drag_scale = (float)flightrl_unpack_number(kwargs, "drag_scale");
    env->randomization_config.thrust_scale = (float)flightrl_unpack_number(kwargs, "thrust_scale");
    env->randomization_config.actuator_tau_scale = (float)flightrl_unpack_number(kwargs, "actuator_tau_scale");
    env->randomization_config.sensor_noise_scale = (float)flightrl_unpack_number(kwargs, "sensor_noise_scale");
    env->wind_config.enabled = (int)flightrl_unpack_number(kwargs, "wind_enabled");
    env->wind_config.steady_x = (float)flightrl_unpack_number(kwargs, "wind_steady_x");
    env->wind_config.steady_z = (float)flightrl_unpack_number(kwargs, "wind_steady_z");
    env->wind_config.gust_strength = (float)flightrl_unpack_number(kwargs, "wind_gust_strength");
    env->wind_config.gust_tau = (float)flightrl_unpack_number(kwargs, "wind_gust_tau");
    env->rng_state = (uint64_t)flightrl_unpack_number(kwargs, "seed") + 0x9e3779b97f4a7c15ULL;

    for (int i = 0; i < FLIGHTRL_MAX_WAYPOINTS; ++i) {
        char key_x[32];
        char key_z[32];
        snprintf(key_x, sizeof(key_x), "waypoint_%d_x", i);
        snprintf(key_z, sizeof(key_z), "waypoint_%d_z", i);
        env->task_config.fixed_waypoints[i].x = (float)flightrl_unpack_number(kwargs, key_x);
        env->task_config.fixed_waypoints[i].z = (float)flightrl_unpack_number(kwargs, key_z);
    }

    if (env->sensor_config.flags & (FLIGHT_OBS_RANGE | FLIGHT_OBS_VISION)) {
        PyErr_SetString(PyExc_NotImplementedError, "range and vision sensors are placeholders in the MVP");
        return -1;
    }
    return PyErr_Occurred() ? -1 : 0;
}

static int my_log(PyObject *dict, Log *log) {
    return flightrl_set_float(dict, "episode_return", log->episode_return) ||
        flightrl_set_float(dict, "episode_length", log->episode_length) ||
        flightrl_set_float(dict, "success_rate", log->success_rate) ||
        flightrl_set_float(dict, "crash_rate", log->crash_rate) ||
        flightrl_set_float(dict, "timeout_rate", log->timeout_rate) ||
        flightrl_set_float(dict, "mean_distance", log->mean_distance) ||
        flightrl_set_float(dict, "mean_action_magnitude", log->mean_action_magnitude);
}

static int my_get(PyObject *dict, DronePlanarEnv *env) {
    TargetPoint target = env->world.targets[env->world.active_target];
    float front_pair = env->motor_thrusts[FLIGHT_ROTOR_FRONT_LEFT] + env->motor_thrusts[FLIGHT_ROTOR_FRONT_RIGHT];
    float rear_pair = env->motor_thrusts[FLIGHT_ROTOR_REAR_LEFT] + env->motor_thrusts[FLIGHT_ROTOR_REAR_RIGHT];
    return flightrl_set_float(dict, "x", env->drone.x) ||
        flightrl_set_float(dict, "z", env->drone.z) ||
        flightrl_set_float(dict, "vx", env->drone.vx) ||
        flightrl_set_float(dict, "vz", env->drone.vz) ||
        flightrl_set_float(dict, "ax", env->last_ax) ||
        flightrl_set_float(dict, "az", env->last_az) ||
        flightrl_set_float(dict, "pitch", env->drone.pitch) ||
        flightrl_set_float(dict, "pitch_rate", env->drone.pitch_rate) ||
        flightrl_set_float(dict, "target_x", target.x) ||
        flightrl_set_float(dict, "target_z", target.z) ||
        flightrl_set_float(dict, "distance", env->current_distance) ||
        flightrl_set_float(dict, "wind_x", env->wind_x) ||
        flightrl_set_float(dict, "wind_z", env->wind_z) ||
        flightrl_set_float(dict, "motor_front_left", env->motor_thrusts[FLIGHT_ROTOR_FRONT_LEFT]) ||
        flightrl_set_float(dict, "motor_front_right", env->motor_thrusts[FLIGHT_ROTOR_FRONT_RIGHT]) ||
        flightrl_set_float(dict, "motor_rear_left", env->motor_thrusts[FLIGHT_ROTOR_REAR_LEFT]) ||
        flightrl_set_float(dict, "motor_rear_right", env->motor_thrusts[FLIGHT_ROTOR_REAR_RIGHT]) ||
        flightrl_set_float(dict, "motor_front_pair", front_pair) ||
        flightrl_set_float(dict, "motor_rear_pair", rear_pair) ||
        flightrl_set_float(dict, "command_0", env->current_action[0]) ||
        flightrl_set_float(dict, "command_1", env->current_action[1]) ||
        flightrl_set_float(dict, "command_2", env->current_action[2]) ||
        flightrl_set_float(dict, "command_3", env->current_action[3]) ||
        flightrl_set_float(dict, "action_dim", (float)env->sensor_config.action_dim) ||
        flightrl_set_float(dict, "active_target", (float)(env->world.active_target + 1)) ||
        flightrl_set_float(dict, "target_count", (float)env->world.target_count) ||
        flightrl_set_float(dict, "reward_total", env->reward_breakdown.total) ||
        flightrl_set_float(dict, "reward_progress", env->reward_breakdown.progress_bonus) ||
        flightrl_set_float(dict, "reward_distance_penalty", env->reward_breakdown.distance_penalty);
}

static PyMethodDef methods[] = {
    {"env_init", (PyCFunction)env_init, METH_VARARGS | METH_KEYWORDS, "Initialize a native environment"},
    {"env_reset", env_reset, METH_VARARGS, "Reset one environment"},
    {"env_step", env_step, METH_VARARGS, "Step one environment"},
    {"env_get", env_get, METH_VARARGS, "Get one environment snapshot"},
    {"env_close", env_close, METH_VARARGS, "Close one environment"},
    {"vectorize", vectorize, METH_VARARGS, "Create a vector handle"},
    {"vec_reset", vec_reset, METH_VARARGS, "Reset all environments"},
    {"vec_step", vec_step, METH_VARARGS, "Step all environments"},
    {"vec_log", vec_log, METH_VARARGS, "Aggregate episode logs"},
    {"vec_close", vec_close, METH_VARARGS, "Close all environments"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_binding",
    NULL,
    -1,
    methods,
};

PyMODINIT_FUNC PyInit__binding(void) {
    import_array();
    return PyModule_Create(&module);
}
