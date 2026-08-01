#ifndef FLIGHTRL_NATIVE_EDGE_STUDENT_ACTION_H
#define FLIGHTRL_NATIVE_EDGE_STUDENT_ACTION_H

void flightrl_edge_student_control_action(
    const float *policy_action,
    float max_yawrate_deg_s,
    float physics_max_yawrate_rad_s,
    float *setpoint,
    float *applied_action
);

#endif
