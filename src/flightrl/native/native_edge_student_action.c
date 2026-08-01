#include <math.h>

#include "native_edge_student_action.h"

#define EDGE_ACTION_PI 3.14159265358979323846f

static float edge_action_clamp(float value) {
    return value < -1.0f ? -1.0f : (value > 1.0f ? 1.0f : value);
}

void flightrl_edge_student_control_action(
    const float *policy_action,
    float max_yawrate_deg_s,
    float physics_max_yawrate_rad_s,
    float *setpoint,
    float *applied_action
) {
    for (int index = 0; index < 4; ++index) {
        applied_action[index] = edge_action_clamp(policy_action[index]);
    }
    setpoint[0] = applied_action[0];
    setpoint[1] = applied_action[1];
    setpoint[2] = applied_action[2];
    float declared_yawrate = max_yawrate_deg_s * EDGE_ACTION_PI / 180.0f;
    setpoint[3] = (
        declared_yawrate > 0.0f && physics_max_yawrate_rad_s > 0.0f
    ) ? edge_action_clamp(
        applied_action[3] * declared_yawrate / physics_max_yawrate_rad_s
    ) : 0.0f;
    applied_action[3] = declared_yawrate > 0.0f
        ? setpoint[3] * physics_max_yawrate_rad_s / declared_yawrate
        : 0.0f;
}
