static float flightrl_puffer_clip_axis(float value, float low, float high, float margin) {
    if (low + margin > high - margin) {
        return 0.5f * (low + high);
    }
    return clampf(value, low + margin, high - margin);
}

static float flightrl_puffer_normal(Env* env, float std) {
    if (std <= 0.0f) {
        return 0.0f;
    }
    float u1 = fmaxf(rnd(&env->rng, 0.0f, 1.0f), 1.0e-7f);
    float u2 = rnd(&env->rng, 0.0f, 1.0f);
    return std * sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

static void flightrl_puffer_euler_quat(float roll, float pitch, float yaw, float* q) {
    float cr = cosf(0.5f * roll);
    float sr = sinf(0.5f * roll);
    float cp = cosf(0.5f * pitch);
    float sp = sinf(0.5f * pitch);
    float cy = cosf(0.5f * yaw);
    float sy = sinf(0.5f * yaw);
    q[0] = cr * cp * cy + sr * sp * sy;
    q[1] = sr * cp * cy - cr * sp * sy;
    q[2] = cr * sp * cy + sr * cp * sy;
    q[3] = cr * cp * sy - sr * sp * cy;
}

static void flightrl_puffer_target(Env* env) {
    int circle = env->target_radius_min >= 0.0f;
    if (circle) {
        float angle = rnd(&env->rng, -3.141592653589793f, 3.141592653589793f);
        float radius = rnd(&env->rng, env->target_radius_min, env->target_radius_max);
        env->target_position[0] = env->position[0] - radius * cosf(angle);
        env->target_position[1] = env->position[1] - radius * sinf(angle);
        env->target_position[2] = rnd(&env->rng, env->target_z_min, env->target_z_max);
    } else if (env->target_xy_offset_abs < 0.0f) {
        env->target_position[0] = rnd(&env->rng, -env->target_xy_abs, env->target_xy_abs);
        env->target_position[1] = rnd(&env->rng, -env->target_xy_abs, env->target_xy_abs);
        env->target_position[2] = rnd(&env->rng, env->target_z_min, env->target_z_max);
    } else {
        env->target_position[0] = env->position[0] + rnd(&env->rng, -env->target_xy_offset_abs, env->target_xy_offset_abs);
        env->target_position[1] = env->position[1] + rnd(&env->rng, -env->target_xy_offset_abs, env->target_xy_offset_abs);
        env->target_position[2] = env->position[2] + rnd(&env->rng, -env->target_z_offset_abs, env->target_z_offset_abs);
    }
    if (env->target_xy_offset_abs == 0.0f && env->target_z_offset_abs == 0.0f) {
        return;
    }
    env->target_position[0] = clampf(env->target_position[0], env->room[0] + 0.25f, env->room[1] - 0.25f);
    env->target_position[1] = clampf(env->target_position[1], env->room[2] + 0.25f, env->room[3] - 0.25f);
    env->target_position[2] = clampf(env->target_position[2], env->room[4] + 0.35f, env->room[5] - 0.25f);
}

static void apply_reset_profile(Env* env) {
    env->position[0] = flightrl_puffer_clip_axis(
        rnd(&env->rng, -env->initial_xy_abs, env->initial_xy_abs), env->room[0], env->room[1], 0.25f
    );
    env->position[1] = flightrl_puffer_clip_axis(
        rnd(&env->rng, -env->initial_xy_abs, env->initial_xy_abs), env->room[2], env->room[3], 0.25f
    );
    env->position[2] = flightrl_puffer_clip_axis(
        rnd(&env->rng, env->reset_z_min, env->reset_z_max), env->room[4], env->room[5], env->reset_z_margin
    );

    float yaw = rnd(&env->rng, -3.141592653589793f, 3.141592653589793f);
    if (env->near_wall_probability > 0.0f && rnd(&env->rng, 0.0f, 1.0f) < env->near_wall_probability) {
        int side = (int)rnd(&env->rng, 0.0f, 4.0f);
        float clearance = rnd(&env->rng, env->near_wall_min_clearance_m, env->near_wall_max_clearance_m);
        float jitter = rnd(&env->rng, -env->near_wall_yaw_jitter_rad, env->near_wall_yaw_jitter_rad);
        if (side == 0) env->position[0] = env->room[1] - clearance;
        else if (side == 1) env->position[0] = env->room[0] + clearance;
        else if (side == 2) env->position[1] = env->room[3] - clearance;
        else env->position[1] = env->room[2] + clearance;
        yaw = jitter;
    }

    float roll = flightrl_puffer_normal(env, env->attitude_std);
    float pitch = flightrl_puffer_normal(env, env->attitude_std);
    flightrl_puffer_target(env);
    if (env->target_radius_min >= 0.0f && env->target_yaw_offset_abs >= 0.0f) {
        yaw = circle_tangent_yaw(env->position, env->target_position) +
            rnd(&env->rng, -env->target_yaw_offset_abs, env->target_yaw_offset_abs);
        yaw = wrap_angle(yaw);
    }
    flightrl_puffer_euler_quat(roll, pitch, yaw, env->quaternion);

    env->velocity[0] = flightrl_puffer_normal(env, env->initial_velocity_xy_std);
    env->velocity[1] = flightrl_puffer_normal(env, env->initial_velocity_xy_std);
    env->velocity[2] = flightrl_puffer_normal(env, env->initial_velocity_z_std);
    if (env->target_yaw_offset_abs < 0.0f) {
        env->target_yaw = rnd(&env->rng, -3.141592653589793f, 3.141592653589793f);
    } else {
        env->target_yaw = wrap_angle(yaw + rnd(&env->rng, -env->target_yaw_offset_abs, env->target_yaw_offset_abs));
    }
    update_ranges(env->position, env->quaternion, env->ranges, env->room);
}
