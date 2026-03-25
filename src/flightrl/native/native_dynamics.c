#include "native_env.h"

void flightrl_apply_dynamics(DronePlanarEnv *env) {
    RuntimeDynamics *dyn = &env->runtime_dynamics;
    DroneState *drone = &env->drone;
    float front_pair = env->motor_thrusts[FLIGHT_ROTOR_FRONT_LEFT] + env->motor_thrusts[FLIGHT_ROTOR_FRONT_RIGHT];
    float rear_pair = env->motor_thrusts[FLIGHT_ROTOR_REAR_LEFT] + env->motor_thrusts[FLIGHT_ROTOR_REAR_RIGHT];
    float total_thrust = front_pair + rear_pair;
    float torque = (rear_pair - front_pair) * dyn->arm_length;
    float sin_pitch = sinf(drone->pitch);
    float cos_pitch = cosf(drone->pitch);
    float rel_vx;
    float rel_vz;

    flightrl_update_wind(env);
    rel_vx = drone->vx - env->wind_x;
    rel_vz = drone->vz - env->wind_z;

    env->last_ax = -(total_thrust / dyn->mass) * sin_pitch - (dyn->drag * rel_vx / dyn->mass);
    env->last_az = (total_thrust / dyn->mass) * cos_pitch - dyn->gravity - (dyn->drag * rel_vz / dyn->mass);

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
