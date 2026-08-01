#ifndef FLIGHTRL_NATIVE_DOOR_MISSION_H
#define FLIGHTRL_NATIVE_DOOR_MISSION_H

typedef struct {
    float target_standoff_m;
    float planar_position_tolerance_m;
    float vertical_position_tolerance_m;
    float standoff_tolerance_m;
    float yaw_alignment_tolerance_rad;
    float max_horizontal_speed_m_s;
    float max_vertical_speed_m_s;
    float max_yaw_rate_rad_s;
    int dwell_steps;
} FlightRLDoorMissionConfig;

typedef struct {
    int dwell_steps;
} FlightRLDoorMissionState;

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
);

#endif
