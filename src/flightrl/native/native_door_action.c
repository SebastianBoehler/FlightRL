#include <math.h>

#include "native_door_action.h"

#define DOOR_ACTION_PI 3.14159265358979323846f

static float door_action_clamp(float value, float lo, float hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

void flightrl_door_control_action(
    const float *policy_action,
    float max_yawrate_deg_s,
    float physics_max_yawrate_rad_s,
    float *setpoint,
    float *executed_previous_action
) {
    float forward = door_action_clamp(policy_action[0], 0.0f, 1.0f);
    float yaw = door_action_clamp(policy_action[1], -1.0f, 1.0f);
    float declared_yawrate = max_yawrate_deg_s * DOOR_ACTION_PI / 180.0f;
    float low_level_yaw = 0.0f;
    if (declared_yawrate > 0.0f && physics_max_yawrate_rad_s > 0.0f) {
        low_level_yaw = door_action_clamp(
            yaw * declared_yawrate / physics_max_yawrate_rad_s,
            -1.0f,
            1.0f
        );
    }
    setpoint[0] = forward;
    setpoint[1] = 0.0f;
    setpoint[2] = 0.0f;
    setpoint[3] = low_level_yaw;
    executed_previous_action[0] = forward;
    executed_previous_action[1] = declared_yawrate > 0.0f
        ? low_level_yaw * physics_max_yawrate_rad_s / declared_yawrate
        : 0.0f;
}
