static void step_one(
    float *p,
    float *v,
    float *q,
    float *rates,
    float *ranges,
    float *thrust_state,
    const float *action,
    const float *physics,
    const float *room,
    float dt
) {
    const float *phys = physics_or_default(physics);
    const float max_rate[3] = {phys[SIXDOF_PHYS_MAX_RATE_ROLL], phys[SIXDOF_PHYS_MAX_RATE_PITCH], phys[SIXDOF_PHYS_MAX_RATE_YAW]};
    float alpha = dt / (phys[SIXDOF_PHYS_RATE_TAU] + dt);
    float target_thrust = 1.0f + phys[SIXDOF_PHYS_THRUST_SCALE] * clampf(action[0], -1.0f, 1.0f);
    float motor_tau = phys[SIXDOF_PHYS_MOTOR_TAU];
    if (motor_tau > 0.0f) {
        float motor_alpha = dt / (motor_tau + dt);
        *thrust_state += motor_alpha * (target_thrust - *thrust_state);
    } else {
        *thrust_state = target_thrust;
    }
    float thrust = phys[SIXDOF_PHYS_MASS] * phys[SIXDOF_PHYS_GRAVITY] * (*thrust_state);

    for (int i = 0; i < 3; ++i) {
        float target = clampf(action[i + 1], -1.0f, 1.0f) * max_rate[i];
        rates[i] += alpha * (target - rates[i]);
    }

    float omega[4] = {0.0f, rates[0], rates[1], rates[2]};
    float qdot[4] = {
        -0.5f * (q[1] * omega[1] + q[2] * omega[2] + q[3] * omega[3]),
        0.5f * (q[0] * omega[1] + q[2] * omega[3] - q[3] * omega[2]),
        0.5f * (q[0] * omega[2] - q[1] * omega[3] + q[3] * omega[1]),
        0.5f * (q[0] * omega[3] + q[1] * omega[2] - q[2] * omega[1]),
    };
    for (int i = 0; i < 4; ++i) {
        q[i] += qdot[i] * dt;
    }
    normalize_quat(q);

    float r[9];
    quat_matrix(q, r);
    float accel[3] = {
        r[2] * thrust / phys[SIXDOF_PHYS_MASS],
        r[5] * thrust / phys[SIXDOF_PHYS_MASS],
        r[8] * thrust / phys[SIXDOF_PHYS_MASS] - phys[SIXDOF_PHYS_GRAVITY],
    };
    for (int i = 0; i < 3; ++i) {
        accel[i] -= phys[SIXDOF_PHYS_LINEAR_DRAG] * v[i];
        v[i] += accel[i] * dt;
        p[i] += v[i] * dt;
    }
    update_ranges(p, q, ranges, room);
}
