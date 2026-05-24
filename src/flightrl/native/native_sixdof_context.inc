static uint32_t rng_next(uint32_t *rng) {
    *rng = 1664525u * (*rng) + 1013904223u;
    return *rng;
}

static float rnd(uint32_t *rng, float lo, float hi) {
    float unit = (float)(rng_next(rng) >> 8) / 16777215.0f;
    return lo + unit * (hi - lo);
}

uint32_t flightrl_sixdof_reset_one(
    float *p, float *v, float *q, float *rates, float *ranges, float *target, float *target_yaw, float *prev,
    int *step_count, float *obs, float *reward, unsigned char *terminal, unsigned char *truncation, const float *room, uint32_t rng
) {
    const float *r = room_or_default(room);
    p[0] = rnd(&rng, -0.8f, 0.8f);
    p[1] = rnd(&rng, -0.8f, 0.8f);
    p[2] = rnd(&rng, 0.35f, 0.9f);
    for (int i = 0; i < 3; ++i) {
        v[i] = 0.0f;
        rates[i] = 0.0f;
        target[i] = rnd(&rng, i == 2 ? 0.45f : -1.0f, i == 2 ? 0.9f : 1.0f);
    }
    float yaw = rnd(&rng, -3.14159265f, 3.14159265f);
    q[0] = cosf(0.5f * yaw);
    q[1] = 0.0f;
    q[2] = 0.0f;
    q[3] = sinf(0.5f * yaw);
    *target_yaw = rnd(&rng, -3.14159265f, 3.14159265f);
    for (int i = 0; i < 4; ++i) {
        prev[i] = 0.0f;
    }
    *step_count = 0;
    update_ranges(p, q, ranges, r);
    assemble_one(p, v, q, rates, ranges, target, *target_yaw, prev, 0, 0, 0, 0.0f, obs, reward, terminal, truncation, r);
    *reward = 0.0f;
    *terminal = 0;
    *truncation = 0;
    return rng;
}

void flightrl_sixdof_step_env_context_batch(
    float *position, float *velocity, float *quaternion, float *body_rates, float *ranges, float *target_position, float *target_yaw,
    float *previous_action, int *step_count, const float *actions, float *observations, float *rewards, unsigned char *terminals,
    unsigned char *truncations, const float *room, const int *task_ids, int reward_mode, const float *previous_error, int num_envs, float dt
) {
    for (int env = 0; env < num_envs; ++env) {
        float *action = previous_action + env * 4;
        for (int i = 0; i < 4; ++i) {
            action[i] = clampf(actions[env * 4 + i], -1.0f, 1.0f);
        }
        step_one(position + env * 3, velocity + env * 3, quaternion + env * 4, body_rates + env * 3, ranges + env * 6, action, room, dt);
        step_count[env] += 1;
        assemble_one(
            position + env * 3, velocity + env * 3, quaternion + env * 4, body_rates + env * 3, ranges + env * 6,
            target_position + env * 3, target_yaw[env], action, step_count[env], task_ids[env], reward_mode, previous_error[env],
            observations + env * SIXDOF_OBS_DIM, rewards + env, terminals + env, truncations + env, room
        );
    }
}
