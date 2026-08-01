static void apply_door_lane_action(Env *env, float *command) {
#ifdef FLIGHTRL_EDGE_STUDENT_LANE
    flightrl_edge_student_control_action(
        env->actions,
        env->max_yawrate_deg_s,
        env->physics[SIXDOF_PHYS_MAX_RATE_YAW],
        command,
        env->previous_action
    );
#else
    flightrl_door_control_action(
        env->actions,
        env->max_yawrate_deg_s,
        env->physics[SIXDOF_PHYS_MAX_RATE_YAW],
        command,
        env->previous_action
    );
#endif
}

static float door_lane_action_cost(const Env *env) {
    float cost = 0.0f;
    for (int index = 0; index < FLIGHTRL_DOOR_ACTION_DIM; ++index) {
        cost += env->previous_action[index] * env->previous_action[index];
    }
    return cost;
}

#ifdef FLIGHTRL_EDGE_STUDENT_LANE
static int write_door_observation(Env *env, uint8_t reset_temporal) {
    float grounding[FLIGHTRL_EDGE_GROUNDING_DIM];
    flightrl_edge_student_observation(
        env->position,
        env->velocity,
        env->quaternion,
        env->body_rates,
        env->room,
        env->scene.door,
        env->scene.obstacle,
        env->camera_mean,
        env->scene_seed,
        env->camera_randomization,
        env->control_step,
        env->camera_mask,
        env->takeoff_origin_z,
        env->origin_position,
        env->origin_yaw,
        env->previous_action,
        grounding,
        env->observations
    );
    /* Bind teacher discovery to the post-occlusion, post-airframe label. */
    int visible = flightrl_edge_student_update_target_observed(
        reset_temporal,
        grounding[0],
        &env->scene.target_observed,
        &env->scene.initial_outside_fov
    );
    flightrl_door_teacher_advance(
        env->position,
        env->quaternion,
        &env->scene
    );
    float teacher[2];
    flightrl_door_teacher_action(
        env->position, env->quaternion, &env->scene,
        env->max_yawrate_deg_s, teacher
    );
    flightrl_edge_student_training_tail(
        teacher,
        grounding,
        reset_temporal ? 0u : env->scene_group_id,
        env->observations
    );
    return visible;
}
#else
static void write_door_observation(Env *env, uint8_t reset_temporal) {
    float grounding[4];
    float proprioception[SIXDOF_DOOR_PROPRIO_DIM] = {0};
    flightrl_door_proprioception(
        env->position, env->velocity, env->quaternion, env->body_rates,
        env->room, env->origin_position, env->origin_yaw, env->previous_action,
        proprioception
    );
    flightrl_sixdof_door_observation_scene(
        env->position,
        env->quaternion,
        env->room,
        env->scene.door,
        env->scene.obstacle,
        env->camera_mean,
        env->scene_seed,
        env->camera_randomization,
        env->previous_frame,
        reset_temporal,
        proprioception,
        grounding,
        env->observations
    );
    if (env->camera_mask > 0.5f) {
        memset(
            env->observations,
            0,
            sizeof(float) * SIXDOF_VISION_CHANNELS * SIXDOF_VISION_PIXELS
        );
    }
    int visible = grounding[0] > 0.5f;
    float detector_grounding[4];
    memcpy(detector_grounding, grounding, sizeof(detector_grounding));
    if (env->camera_mask > 0.5f) {
        memset(detector_grounding, 0, sizeof(detector_grounding));
    }
    flightrl_door_detector_update(
        &env->detector, detector_grounding, env->control_step,
        &env->appearance_rng, env->control_dt, env->maximum_evidence_age_s
    );
    int detected = (
        env->detector.evidence[0] > 0.0f
        && env->detector.evidence[4] < 1.0f
    );
    int phase = detected
        ? (env->detector.evidence[3] >= 0.55f ? 2 : 1)
        : (env->detector.target_seen ? 3 : 0);
    int phase_offset = (
        SIXDOF_VISION_CHANNELS * SIXDOF_VISION_PIXELS
        + SIXDOF_DOOR_SENSOR_DIM
    );
    env->observations[phase_offset + phase] = 1.0f;
    memcpy(
        env->observations + phase_offset + SIXDOF_DOOR_PHASE_DIM,
        env->detector.evidence,
        sizeof(env->detector.evidence)
    );
    float teacher[2];
    flightrl_door_teacher_action(
        env->position, env->quaternion, &env->scene,
        env->max_yawrate_deg_s, teacher
    );
    env->observations[SIXDOF_DOOR_POLICY_OBS_DIM] = teacher[0];
    env->observations[SIXDOF_DOOR_POLICY_OBS_DIM + 1] = teacher[1];
    env->observations[SIXDOF_DOOR_POLICY_OBS_DIM + 2] = (float)visible;
    env->observations[SIXDOF_DOOR_POLICY_OBS_DIM + 3] = grounding[1];
    env->observations[SIXDOF_DOOR_POLICY_OBS_DIM + 4] = grounding[2];
    env->observations[SIXDOF_DOOR_POLICY_OBS_DIM + 5] = grounding[3];
}
#endif
