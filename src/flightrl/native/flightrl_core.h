#ifndef FLIGHTRL_CORE_H
#define FLIGHTRL_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FLIGHTRL_CORE_ABI_VERSION 1u

enum FlightRLCoreStatus {
    FLIGHTRL_CORE_OK = 0,
    FLIGHTRL_CORE_INVALID_ARGUMENT = 1,
    FLIGHTRL_CORE_INCOMPATIBLE_ABI = 2,
};

/*
 * Stable host seam for one contiguous batched six-DoF dynamics step.
 * Shapes are (N,3), (N,3), (N,4), (N,3), (N,6), (N), (N,4),
 * (N,SIXDOF_PHYSICS_DIM), and (7). All floats are native-endian float32.
 */
typedef struct FlightRLSixDofBatch {
    uint32_t abi_version;
    size_t struct_size;
    int32_t num_envs;
    float dt_s;
    float *position_m;
    float *velocity_m_s;
    float *quaternion_wxyz;
    float *body_rates_rad_s;
    float *ranges_m;
    float *thrust_state;
    const float *actions_normalized;
    const float *physics_parameters;
    const float *room_bounds_m;
} FlightRLSixDofBatch;

typedef struct FlightRLSixDofEnvironmentBatch {
    FlightRLSixDofBatch dynamics;
    float *target_position_m;
    float *target_yaw_rad;
    float *previous_action_normalized;
    int32_t *step_count;
    float *observations;
    float *rewards;
    uint8_t *terminals;
    uint8_t *truncations;
    const int32_t *task_ids;
    int32_t reward_mode;
    const float *previous_error;
} FlightRLSixDofEnvironmentBatch;

uint32_t flightrl_core_abi_version(void);
int flightrl_core_step_sixdof(FlightRLSixDofBatch *batch);
int flightrl_core_step_environment(FlightRLSixDofEnvironmentBatch *batch);
int flightrl_core_step_environment_with_context(
    FlightRLSixDofEnvironmentBatch *batch
);

#ifdef __cplusplus
}
#endif

#endif
