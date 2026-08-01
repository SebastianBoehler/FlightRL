static void capture_door_episode_group(Env *env, uint8_t low_light) {
    env->scene_group_id = flightrl_door_scene_group_id(
        env->layout_family,
        (uint8_t)env->scene.door[0],
        low_light,
        env->scene.detour_active,
        env->scene.initial_outside_fov
    );
}

static void log_door_episode_group(Env *env, uint8_t success) {
    flightrl_door_group_log_add(
        &env->log.groups,
        env->scene_group_id,
        success
    );
}
