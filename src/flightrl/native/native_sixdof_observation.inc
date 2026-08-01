static float yaw_from_quat(const float *q) {
    return atan2f(2.0f * (q[0] * q[3] + q[1] * q[2]), 1.0f - 2.0f * (q[2] * q[2] + q[3] * q[3]));
}

static float wrap_angle(float value) {
    const float two_pi = 6.283185307179586f;
    while (value > 3.141592653589793f) {
        value -= two_pi;
    }
    while (value < -3.141592653589793f) {
        value += two_pi;
    }
    return value;
}

static float circle_tangent_yaw(const float *p, const float *target) {
    float rx = p[0] - target[0];
    float ry = p[1] - target[1];
    float radius = sqrtf(rx * rx + ry * ry);
    if (radius < 0.2f) {
        radius = 0.2f;
    }
    return atan2f(rx / radius, -ry / radius);
}

static float task_yaw_target(const float *p, const float *target, float target_yaw, int task_id) {
    return task_id == SIXDOF_TASK_CIRCLE ? circle_tangent_yaw(p, target) : target_yaw;
}

static float norm3(const float *values);

static float task_position_error(const float *p, const float *target, int task_id) {
    if (task_id == SIXDOF_TASK_CIRCLE) {
        float dx = p[0] - target[0];
        float dy = p[1] - target[1];
        float radius_error = sqrtf(dx * dx + dy * dy) - 0.75f;
        float z_error = p[2] - target[2];
        return sqrtf(radius_error * radius_error + z_error * z_error);
    }
    float err[3] = {target[0] - p[0], target[1] - p[1], target[2] - p[2]};
    return norm3(err);
}

static unsigned char in_room(const float *p, const float *room) {
    const float *r = room_or_default(room);
    const float margin = 0.03f;
    return p[0] >= r[0] + margin && p[0] <= r[1] - margin && p[1] >= r[2] + margin && p[1] <= r[3] - margin &&
        p[2] >= r[4] + margin && p[2] <= r[5] - margin;
}

static float norm3(const float *values) {
    return sqrtf(values[0] * values[0] + values[1] * values[1] + values[2] * values[2]);
}

static void assemble_one(
    const float *p,
    const float *v,
    const float *q,
    const float *rates,
    const float *ranges,
    const float *target,
    float target_yaw,
    const float *action,
    int step_count,
    int task_id,
    int reward_mode,
    float previous_error,
    const float *physics,
    float *obs,
    float *reward,
    unsigned char *terminal,
    unsigned char *truncation,
    const float *room
) {
    const float *phys = physics_or_default(physics);
    float pos_error[3] = {target[0] - p[0], target[1] - p[1], target[2] - p[2]};
    float effective_yaw = task_yaw_target(p, target, target_yaw, task_id);
    float yaw_error = wrap_angle(effective_yaw - yaw_from_quat(q));
    float min_clearance = ranges[0];
    for (int i = 1; i < 4; ++i) {
        if (ranges[i] < min_clearance) {
            min_clearance = ranges[i];
        }
    }
    float clearance_penalty = fmaxf(0.0f, 0.35f - min_clearance);
    float action_norm = sqrtf(action[0] * action[0] + action[1] * action[1] + action[2] * action[2] + action[3] * action[3]);
    float current_error = task_position_error(p, target, task_id);
    *reward = 1.0f - current_error - 0.15f * norm3(v) - 0.1f * fabsf(yaw_error) - 1.5f * clearance_penalty - 0.02f * action_norm;
    *terminal = in_room(p, room) ? 0 : 1;
    *truncation = step_count >= 800 ? 1 : 0;
    if (reward_mode >= SIXDOF_REWARD_PROGRESS && reward_mode <= SIXDOF_REWARD_LIVE_STABLE_CLEARANCE) {
        int stable = reward_mode == SIXDOF_REWARD_LIVE_STABLE_CLEARANCE;
        int live = reward_mode == SIXDOF_REWARD_LIVE_CLEARANCE || stable;
        float threshold = reward_mode == SIXDOF_REWARD_PROGRESS ? 0.25f : (live ? 0.65f : 0.45f);
        float clearance_weight = reward_mode == SIXDOF_REWARD_PROGRESS ? 1.0f : (live ? 5.0f : 2.5f);
        float yaw_weight = reward_mode == SIXDOF_REWARD_PROGRESS_YAW_CLEARANCE ? 0.6f : 0.1f;
        float progress_clearance = fmaxf(0.0f, threshold - min_clearance);
        float clearance_bonus = live ? 0.15f * fminf(min_clearance, threshold) : 0.0f;
        float open_space = stable ? clampf((min_clearance - 0.45f) / 0.20f, 0.0f, 1.0f) : 0.0f;
        float horizontal_speed = sqrtf(v[0] * v[0] + v[1] * v[1]);
        float tilt_action = sqrtf(action[1] * action[1] + action[2] * action[2]);
        float stable_penalty = open_space * (0.35f * horizontal_speed + 0.10f * tilt_action);
        float escape_reward = 0.0f;
        float vertical_reward = 0.0f;
        if (stable) {
            float push_x = fmaxf(0.0f, threshold - ranges[1]) - fmaxf(0.0f, threshold - ranges[0]);
            float push_y = fmaxf(0.0f, threshold - ranges[3]) - fmaxf(0.0f, threshold - ranges[2]);
            float rmat[9];
            quat_matrix(q, rmat);
            float body_vx = rmat[0] * v[0] + rmat[3] * v[1] + rmat[6] * v[2];
            float body_vy = rmat[1] * v[0] + rmat[4] * v[1] + rmat[7] * v[2];
            float velocity_alignment = push_x * body_vx + push_y * body_vy;
            float action_alignment = push_x * action[2] - push_y * action[1];
            escape_reward = 1.20f * velocity_alignment + 0.80f * action_alignment;
            float top_pressure = fmaxf(0.0f, 0.45f - ranges[4]);
            float bottom_pressure = fmaxf(0.0f, 0.45f - ranges[5]);
            float vertical_push = bottom_pressure - top_pressure;
            vertical_reward = -3.00f * (top_pressure + bottom_pressure) + 0.50f * vertical_push * v[2] + 0.25f * vertical_push * action[0];
        }
        float speed_weight = stable ? 0.12f : 0.02f;
        float terminal_penalty = stable ? 2.0f : 1.0f;
        *reward = 0.2f + 3.0f * (previous_error - current_error) - 0.05f * current_error - speed_weight * norm3(v) -
            yaw_weight * fabsf(yaw_error) - clearance_weight * progress_clearance + clearance_bonus - 0.01f * action_norm -
            stable_penalty + escape_reward + vertical_reward - terminal_penalty * ((*terminal || *truncation) ? 1.0f : 0.0f);
    }
    obs[0] = pos_error[0] / 2.0f;
    obs[1] = pos_error[1] / 2.0f;
    obs[2] = pos_error[2] / 1.5f;
    obs[3] = v[0] / 3.0f;
    obs[4] = v[1] / 3.0f;
    obs[5] = v[2] / 3.0f;
    for (int i = 0; i < 4; ++i) {
        obs[6 + i] = q[i];
    }
    obs[10] = rates[0] / phys[SIXDOF_PHYS_MAX_RATE_ROLL];
    obs[11] = rates[1] / phys[SIXDOF_PHYS_MAX_RATE_PITCH];
    obs[12] = rates[2] / phys[SIXDOF_PHYS_MAX_RATE_YAW];
    obs[13] = target[0] / 2.0f;
    obs[14] = target[1] / 2.0f;
    obs[15] = target[2] / 2.5f;
    obs[16] = sinf(yaw_error);
    obs[17] = cosf(yaw_error);
    for (int i = 0; i < 6; ++i) {
        obs[18 + i] = ranges[i] / room_or_default(room)[6];
    }
    for (int i = 0; i < 4; ++i) {
        obs[24 + i] = action[i];
    }
}
