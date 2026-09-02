#ifndef FLIGHTRL_BINDING_SIXDOF_CORE_H
#define FLIGHTRL_BINDING_SIXDOF_CORE_H

#include "binding_helpers.h"
#include "flightrl_core.h"


static FlightRLSixDofBatch sixdof_core_batch(
    PyArrayObject *pos,
    PyArrayObject *vel,
    PyArrayObject *quat,
    PyArrayObject *rates,
    PyArrayObject *ranges,
    PyArrayObject *thrust_state,
    PyArrayObject *actions,
    PyArrayObject *physics,
    PyArrayObject *room,
    int num_envs,
    float dt
) {
    return (FlightRLSixDofBatch) {
        .abi_version = FLIGHTRL_CORE_ABI_VERSION,
        .struct_size = sizeof(FlightRLSixDofBatch),
        .num_envs = num_envs,
        .dt_s = dt,
        .position_m = PyArray_DATA(pos),
        .velocity_m_s = PyArray_DATA(vel),
        .quaternion_wxyz = PyArray_DATA(quat),
        .body_rates_rad_s = PyArray_DATA(rates),
        .ranges_m = PyArray_DATA(ranges),
        .thrust_state = PyArray_DATA(thrust_state),
        .actions_normalized = PyArray_DATA(actions),
        .physics_parameters = PyArray_DATA(physics),
        .room_bounds_m = PyArray_DATA(room),
    };
}


static FlightRLSixDofEnvironmentBatch sixdof_core_environment_batch(
    FlightRLSixDofBatch dynamics,
    PyArrayObject *target,
    PyArrayObject *target_yaw,
    PyArrayObject *previous_action,
    PyArrayObject *step_count,
    PyArrayObject *observations,
    PyArrayObject *rewards,
    PyArrayObject *terminals,
    PyArrayObject *truncations
) {
    return (FlightRLSixDofEnvironmentBatch) {
        .dynamics = dynamics,
        .target_position_m = PyArray_DATA(target),
        .target_yaw_rad = PyArray_DATA(target_yaw),
        .previous_action_normalized = PyArray_DATA(previous_action),
        .step_count = PyArray_DATA(step_count),
        .observations = PyArray_DATA(observations),
        .rewards = PyArray_DATA(rewards),
        .terminals = PyArray_DATA(terminals),
        .truncations = PyArray_DATA(truncations),
    };
}

#endif
