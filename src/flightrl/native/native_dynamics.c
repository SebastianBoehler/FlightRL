#include "native_env.h"

void flightrl_apply_dynamics(DronePlanarEnv *env) {
    RuntimeDynamics *dyn = &env->runtime_dynamics;
    DroneState *drone = &env->drone;

    float total_thrust = env->motor_thrusts[0] + env->motor_thrusts[1];
    float torque = (env->motor_thrusts[1] - env->motor_thrusts[0]) * dyn->arm_length;
    float sin_pitch = sinf(drone->pitch);
    float cos_pitch = cosf(drone->pitch);

    env->last_ax = -(total_thrust / dyn->mass) * sin_pitch - (dyn->drag * drone->vx / dyn->mass);
    env->last_az = (total_thrust / dyn->mass) * cos_pitch - dyn->gravity - (dyn->drag * drone->vz / dyn->mass);

    float pitch_acc = (torque - (dyn->angular_drag * drone->pitch_rate)) / dyn->inertia;

    drone->vx += env->last_ax * env->dt;
    drone->vz += env->last_az * env->dt;
    drone->pitch_rate += pitch_acc * env->dt;

    drone->vx = flightrl_clamp(drone->vx, -dyn->max_velocity, dyn->max_velocity);
    drone->vz = flightrl_clamp(drone->vz, -dyn->max_velocity, dyn->max_velocity);
    drone->pitch_rate = flightrl_clamp(drone->pitch_rate, -dyn->max_pitch_rate, dyn->max_pitch_rate);

    drone->x += drone->vx * env->dt;
    drone->z += drone->vz * env->dt;
    drone->pitch += drone->pitch_rate * env->dt;
}
