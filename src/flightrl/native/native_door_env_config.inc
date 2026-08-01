void my_init(Env *env, Dict *kwargs) {
    env->num_agents = 1;
    uint32_t env_index = env->rng;
    uint32_t seed = (uint32_t)dict_get(kwargs, "seed")->value;
    uint32_t appearance_seed = (uint32_t)dict_get(
        kwargs,
        "appearance_seed"
    )->value;
    flightrl_door_episode_rng_init(
        &env->episode_rng,
        seed,
        appearance_seed,
        env_index
    );
    env->control_dt = (float)dict_get(kwargs, "control_dt")->value;
    env->maximum_evidence_age_s = (float)dict_get(
        kwargs,
        "maximum_evidence_age_s"
    )->value;
    env->physics_substeps = (int)dict_get(kwargs, "physics_substeps")->value;
    env->max_episode_steps = (int)dict_get(kwargs, "max_episode_steps")->value;
    env->mission.target_standoff_m = (float)dict_get(
        kwargs, "mission_target_standoff_m"
    )->value;
    env->mission.planar_position_tolerance_m = (float)dict_get(
        kwargs, "mission_planar_position_tolerance_m"
    )->value;
    env->mission.vertical_position_tolerance_m = (float)dict_get(
        kwargs, "mission_vertical_position_tolerance_m"
    )->value;
    env->mission.standoff_tolerance_m = (float)dict_get(
        kwargs, "mission_standoff_tolerance_m"
    )->value;
    env->mission.yaw_alignment_tolerance_rad = (float)dict_get(
        kwargs, "mission_yaw_alignment_tolerance_rad"
    )->value;
    env->mission.max_horizontal_speed_m_s = (float)dict_get(
        kwargs, "mission_max_horizontal_speed_m_s"
    )->value;
    env->mission.max_vertical_speed_m_s = (float)dict_get(
        kwargs, "mission_max_vertical_speed_m_s"
    )->value;
    env->mission.max_yaw_rate_rad_s = (float)dict_get(
        kwargs, "mission_max_yaw_rate_rad_s"
    )->value;
    env->mission.dwell_steps = (int)dict_get(
        kwargs, "mission_dwell_steps"
    )->value;
    env->max_horizontal_speed = (float)dict_get(
        kwargs,
        "max_horizontal_speed_m_s"
    )->value;
    env->max_vertical_speed = (float)dict_get(
        kwargs,
        "max_vertical_speed_m_s"
    )->value;
    env->max_yawrate_deg_s = (float)dict_get(
        kwargs,
        "max_yawrate_deg_s"
    )->value;
    env->velocity_gain = (float)dict_get(kwargs, "velocity_gain")->value;
    env->attitude_gain = (float)dict_get(kwargs, "attitude_gain")->value;
    env->vertical_gain = (float)dict_get(kwargs, "vertical_gain")->value;
    env->camera_mean_min = (float)dict_get(kwargs, "camera_mean_min")->value;
    env->camera_mean_max = (float)dict_get(kwargs, "camera_mean_max")->value;
    env->domain_randomization = (float)dict_get(
        kwargs,
        "domain_randomization"
    )->value;
    env->layout_diversity = (float)dict_get(
        kwargs,
        "layout_diversity"
    )->value;
    env->camera_randomization = (float)dict_get(
        kwargs,
        "camera_randomization"
    )->value;
    env->obstacle_probability = (float)dict_get(
        kwargs,
        "obstacle_probability"
    )->value;
    env->camera_mask = (float)dict_get(kwargs, "camera_mask")->value;
    const char *room_keys[7] = {
        "room_x_min", "room_x_max", "room_y_min", "room_y_max",
        "room_z_min", "room_z_max", "max_range_m"
    };
    const char *physics_keys[9] = {
        "mass_kg", "gravity_m_s2", "linear_drag", "rate_tau_s",
        "thrust_scale", "max_rate_roll", "max_rate_pitch",
        "max_rate_yaw", "motor_tau_s"
    };
    for (int i = 0; i < 7; ++i) {
        env->base_room[i] = (float)dict_get(kwargs, room_keys[i])->value;
    }
    for (int i = 0; i < 9; ++i) {
        env->base_physics[i] = (float)dict_get(
            kwargs,
            physics_keys[i]
        )->value;
    }
}

void my_log(Log *log, Dict *out) {
    static const char *layout_episode_keys[3] = {
        "layout_family_1_episode_fraction",
        "layout_family_2_episode_fraction",
        "layout_family_3_episode_fraction",
    };
    static const char *layout_success_keys[3] = {
        "layout_family_1_success_fraction",
        "layout_family_2_success_fraction",
        "layout_family_3_success_fraction",
    };
    static const char *face_episode_keys[3] = {
        "door_face_1_episode_fraction",
        "door_face_2_episode_fraction",
        "door_face_3_episode_fraction",
    };
    static const char *face_success_keys[3] = {
        "door_face_1_success_fraction",
        "door_face_2_success_fraction",
        "door_face_3_success_fraction",
    };
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "success_rate", log->success_rate);
    dict_set(out, "collision_rate", log->collision_rate);
    dict_set(
        out,
        "outside_fov_success_fraction",
        log->outside_fov_success_fraction
    );
    dict_set(
        out,
        "outside_fov_episode_fraction",
        log->outside_fov_episode_fraction
    );
    dict_set(
        out,
        "outside_fov_observed_fraction",
        log->outside_fov_observed_fraction
    );
    dict_set(
        out,
        "observed_episode_fraction",
        log->observed_episode_fraction
    );
    dict_set(out, "door_visible_fraction", log->door_visible_fraction);
    for (int i = 0; i < 3; ++i) {
        dict_set(out, layout_episode_keys[i], log->groups.layout_episode_fraction[i]);
        dict_set(out, layout_success_keys[i], log->groups.layout_success_fraction[i]);
        dict_set(out, face_episode_keys[i], log->groups.door_face_episode_fraction[i]);
        dict_set(out, face_success_keys[i], log->groups.door_face_success_fraction[i]);
    }
    dict_set(
        out,
        "low_light_episode_fraction",
        log->groups.low_light_episode_fraction
    );
    dict_set(
        out,
        "low_light_success_fraction",
        log->groups.low_light_success_fraction
    );
    dict_set(
        out,
        "obstacle_episode_fraction",
        log->groups.obstacle_episode_fraction
    );
    dict_set(
        out,
        "obstacle_success_fraction",
        log->groups.obstacle_success_fraction
    );
    dict_set(
        out,
        "scene_group_schema_version",
        FLIGHTRL_DOOR_GROUP_SCHEMA_VERSION
    );
}
