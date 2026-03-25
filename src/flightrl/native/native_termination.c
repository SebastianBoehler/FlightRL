#include "native_env.h"

int flightrl_check_termination(DronePlanarEnv *env) {
    RuntimeDynamics *dyn = &env->runtime_dynamics;

    if (env->drone.z <= dyn->floor_z) {
        return FLIGHT_TERM_CRASH;
    }

    if (fabsf(env->drone.x) > dyn->x_limit || env->drone.z > dyn->z_limit) {
        return FLIGHT_TERM_OUT_OF_BOUNDS;
    }

    if (fabsf(env->drone.pitch) > dyn->max_pitch_angle) {
        return FLIGHT_TERM_CRASH;
    }

    if ((int)env->step_count >= env->task_config.max_steps) {
        return FLIGHT_TERM_TIMEOUT;
    }

    return FLIGHT_TERM_NONE;
}
