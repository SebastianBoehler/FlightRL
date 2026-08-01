#ifndef FLIGHTRL_NATIVE_DOOR_ACTION_H
#define FLIGHTRL_NATIVE_DOOR_ACTION_H

void flightrl_door_control_action(
    const float *policy_action,
    float max_yawrate_deg_s,
    float physics_max_yawrate_rad_s,
    float *setpoint,
    float *executed_previous_action
);

#endif
