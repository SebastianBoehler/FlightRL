#include <math.h>
#include <stddef.h>

#include "flightrl_core.h"
#include "native_sixdof.h"


uint32_t flightrl_core_abi_version(void) {
    return FLIGHTRL_CORE_ABI_VERSION;
}


static int validate_dynamics_batch(const FlightRLSixDofBatch *batch) {
    if (batch == NULL) {
        return FLIGHTRL_CORE_INVALID_ARGUMENT;
    }
    if (
        batch->abi_version != FLIGHTRL_CORE_ABI_VERSION ||
        batch->struct_size != sizeof(FlightRLSixDofBatch)
    ) {
        return FLIGHTRL_CORE_INCOMPATIBLE_ABI;
    }
    if (
        batch->num_envs <= 0 ||
        !isfinite(batch->dt_s) ||
        batch->dt_s <= 0.0f ||
        batch->position_m == NULL ||
        batch->velocity_m_s == NULL ||
        batch->quaternion_wxyz == NULL ||
        batch->body_rates_rad_s == NULL ||
        batch->ranges_m == NULL ||
        batch->thrust_state == NULL ||
        batch->actions_normalized == NULL ||
        batch->physics_parameters == NULL ||
        batch->room_bounds_m == NULL
    ) {
        return FLIGHTRL_CORE_INVALID_ARGUMENT;
    }

    return FLIGHTRL_CORE_OK;
}


static int validate_environment_batch(
    const FlightRLSixDofEnvironmentBatch *batch
) {
    int status;
    if (batch == NULL) {
        return FLIGHTRL_CORE_INVALID_ARGUMENT;
    }
    status = validate_dynamics_batch(&batch->dynamics);
    if (status != FLIGHTRL_CORE_OK) {
        return status;
    }
    if (
        batch->target_position_m == NULL ||
        batch->target_yaw_rad == NULL ||
        batch->previous_action_normalized == NULL ||
        batch->step_count == NULL ||
        batch->observations == NULL ||
        batch->rewards == NULL ||
        batch->terminals == NULL ||
        batch->truncations == NULL
    ) {
        return FLIGHTRL_CORE_INVALID_ARGUMENT;
    }
    return FLIGHTRL_CORE_OK;
}


int flightrl_core_step_sixdof(FlightRLSixDofBatch *batch) {
    int status = validate_dynamics_batch(batch);
    if (status != FLIGHTRL_CORE_OK) {
        return status;
    }
    flightrl_sixdof_step_batch(
        batch->position_m,
        batch->velocity_m_s,
        batch->quaternion_wxyz,
        batch->body_rates_rad_s,
        batch->ranges_m,
        batch->thrust_state,
        batch->actions_normalized,
        batch->physics_parameters,
        batch->room_bounds_m,
        batch->num_envs,
        batch->dt_s
    );
    return FLIGHTRL_CORE_OK;
}


int flightrl_core_step_environment(FlightRLSixDofEnvironmentBatch *batch) {
    int status = validate_environment_batch(batch);
    if (status != FLIGHTRL_CORE_OK) {
        return status;
    }
    FlightRLSixDofBatch *state = &batch->dynamics;
    flightrl_sixdof_step_env_batch(
        state->position_m,
        state->velocity_m_s,
        state->quaternion_wxyz,
        state->body_rates_rad_s,
        state->ranges_m,
        state->thrust_state,
        state->physics_parameters,
        batch->target_position_m,
        batch->target_yaw_rad,
        batch->previous_action_normalized,
        batch->step_count,
        state->actions_normalized,
        batch->observations,
        batch->rewards,
        batch->terminals,
        batch->truncations,
        state->room_bounds_m,
        state->num_envs,
        state->dt_s
    );
    return FLIGHTRL_CORE_OK;
}


int flightrl_core_step_environment_with_context(
    FlightRLSixDofEnvironmentBatch *batch
) {
    int status = validate_environment_batch(batch);
    if (status != FLIGHTRL_CORE_OK) {
        return status;
    }
    if (batch->task_ids == NULL || batch->previous_error == NULL) {
        return FLIGHTRL_CORE_INVALID_ARGUMENT;
    }
    FlightRLSixDofBatch *state = &batch->dynamics;
    flightrl_sixdof_step_env_context_batch(
        state->position_m,
        state->velocity_m_s,
        state->quaternion_wxyz,
        state->body_rates_rad_s,
        state->ranges_m,
        state->thrust_state,
        state->physics_parameters,
        batch->target_position_m,
        batch->target_yaw_rad,
        batch->previous_action_normalized,
        batch->step_count,
        state->actions_normalized,
        batch->observations,
        batch->rewards,
        batch->terminals,
        batch->truncations,
        state->room_bounds_m,
        batch->task_ids,
        batch->reward_mode,
        batch->previous_error,
        state->num_envs,
        state->dt_s
    );
    return FLIGHTRL_CORE_OK;
}
