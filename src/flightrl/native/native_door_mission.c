#include <math.h>

#include "native_door_mission.h"

#define MISSION_PI 3.14159265358979323846f

static float mission_yaw(const float *quaternion) {
    return atan2f(
        2.0f * (
            quaternion[0] * quaternion[3]
            + quaternion[1] * quaternion[2]
        ),
        1.0f - 2.0f * (
            quaternion[2] * quaternion[2]
            + quaternion[3] * quaternion[3]
        )
    );
}

static float wrap_mission_angle(float value) {
    while (value > MISSION_PI) value -= 2.0f * MISSION_PI;
    while (value < -MISSION_PI) value += 2.0f * MISSION_PI;
    return value;
}

static float door_standoff(
    const float *position,
    const float *room,
    int door_face
) {
    if (door_face == 0) return position[0] - room[0];
    if (door_face == 1) return room[1] - position[0];
    if (door_face == 2) return position[1] - room[2];
    return room[3] - position[1];
}

static int finite_vector(const float *values, int count) {
    for (int i = 0; i < count; ++i) {
        if (!isfinite(values[i])) return 0;
    }
    return 1;
}

static int valid_config(const FlightRLDoorMissionConfig *config) {
    const float values[8] = {
        config->target_standoff_m,
        config->planar_position_tolerance_m,
        config->vertical_position_tolerance_m,
        config->standoff_tolerance_m,
        config->yaw_alignment_tolerance_rad,
        config->max_horizontal_speed_m_s,
        config->max_vertical_speed_m_s,
        config->max_yaw_rate_rad_s,
    };
    if (config->dwell_steps <= 0) return 0;
    for (int i = 0; i < 8; ++i) {
        if (!isfinite(values[i]) || values[i] <= 0.0f) return 0;
    }
    return 1;
}

int flightrl_door_mission_step(
    const FlightRLDoorMissionConfig *config,
    FlightRLDoorMissionState *state,
    const float *position,
    const float *velocity,
    const float *quaternion,
    const float *body_rates,
    const float *room,
    int door_face,
    const float *target_position,
    float target_yaw,
    int visible
) {
    int valid = (
        config != 0 && state != 0
        && position != 0 && velocity != 0 && quaternion != 0
        && body_rates != 0 && room != 0 && target_position != 0
        && door_face >= 0 && door_face < 4
        && valid_config(config)
        && finite_vector(position, 3)
        && finite_vector(velocity, 3)
        && finite_vector(quaternion, 4)
        && finite_vector(body_rates, 3)
        && finite_vector(room, 7)
        && finite_vector(target_position, 3)
        && isfinite(target_yaw)
        && (visible == 0 || visible == 1)
        && state->dwell_steps >= 0
        && state->dwell_steps <= config->dwell_steps
    );
    if (!valid) {
        if (state != 0) state->dwell_steps = 0;
        return 0;
    }

    float dx = target_position[0] - position[0];
    float dy = target_position[1] - position[1];
    float planar_error = sqrtf(dx * dx + dy * dy);
    float horizontal_speed = sqrtf(
        velocity[0] * velocity[0] + velocity[1] * velocity[1]
    );
    int in_tolerance = (
        visible != 0
        && planar_error <= config->planar_position_tolerance_m
        && fabsf(target_position[2] - position[2])
            <= config->vertical_position_tolerance_m
        && fabsf(
            door_standoff(position, room, door_face)
            - config->target_standoff_m
        ) <= config->standoff_tolerance_m
        && fabsf(wrap_mission_angle(mission_yaw(quaternion) - target_yaw))
            <= config->yaw_alignment_tolerance_rad
        && horizontal_speed <= config->max_horizontal_speed_m_s
        && fabsf(velocity[2]) <= config->max_vertical_speed_m_s
        && fabsf(body_rates[2]) <= config->max_yaw_rate_rad_s
    );
    if (!in_tolerance) {
        state->dwell_steps = 0;
        return 0;
    }
    if (state->dwell_steps < config->dwell_steps) state->dwell_steps += 1;
    return state->dwell_steps >= config->dwell_steps;
}
