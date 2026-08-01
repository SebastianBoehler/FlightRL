static void randomize_door_domain(Env *env) {
    float strength = clampf(env->domain_randomization, 0.0f, 1.0f);
    memcpy(env->room, env->base_room, sizeof(env->room));
    memcpy(env->physics, env->base_physics, sizeof(env->physics));
    float x_scale = 1.0f + strength * rnd(&env->rng, -0.20f, 0.20f);
    float y_scale = 1.0f + strength * rnd(&env->rng, -0.20f, 0.20f);
    env->layout_family = 0;
    if (env->layout_diversity > 0.5f) {
        int family = (int)rnd(&env->rng, 0.0f, 3.999999f);
        env->layout_family = (uint8_t)family;
        if (family == 1) {
            x_scale = rnd(&env->rng, 1.20f, 1.50f);
            y_scale = rnd(&env->rng, 0.72f, 0.96f);
        } else if (family == 2) {
            x_scale = rnd(&env->rng, 0.72f, 0.96f);
            y_scale = rnd(&env->rng, 1.20f, 1.50f);
        } else if (family == 3) {
            x_scale = rnd(&env->rng, 0.72f, 0.90f);
            y_scale = rnd(&env->rng, 0.72f, 0.90f);
        }
    }
    float x_half = env->base_room[1] * x_scale;
    float y_half = env->base_room[3] * y_scale;
    env->room[0] = -x_half;
    env->room[1] = x_half;
    env->room[2] = -y_half;
    env->room[3] = y_half;
    env->room[5] *= 1.0f + strength * rnd(&env->rng, -0.10f, 0.10f);
    env->physics[0] *= 1.0f + strength * rnd(&env->rng, -0.08f, 0.08f);
    env->physics[2] *= 1.0f + strength * rnd(&env->rng, -0.30f, 0.30f);
    env->physics[3] *= 1.0f + strength * rnd(&env->rng, -0.15f, 0.15f);
    env->physics[4] *= 1.0f + strength * rnd(&env->rng, -0.08f, 0.08f);
}
