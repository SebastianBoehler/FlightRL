#include <math.h>
#include <stdint.h>
#include <string.h>

#include "native_sixdof.h"
#include "native_sixdof.c"
#include "native_sixdof_setpoint.h"
#include "native_sixdof_setpoint.c"
#include "native_door_action.h"
#include "native_door_action.c"
#include "native_door_mission.h"
#include "native_door_mission.c"
#include "native_door_episode_rng.c"
#include "native_door_proprio.h"
#include "native_door_proprio.c"
#include "native_door_detector.c"
#include "native_door_self_mask.c"
#include "native_sixdof_vision.h"
#include "native_sixdof_vision.c"
#include "native_door_scene.h"
#include "native_door_scene.c"
#include "native_door_teacher.h"
#include "native_door_teacher.c"

#ifdef FLIGHTRL_EDGE_STUDENT_LANE
#include "native_edge_student_action.h"
#include "native_edge_student_action.c"
#include "native_edge_student_observation.h"
#include "native_edge_student_observation.c"
#define OBS_SIZE FLIGHTRL_EDGE_STUDENT_OBS_DIM
#define NUM_ATNS 4
#define ACT_SIZES {1, 1, 1, 1}
#define FLIGHTRL_DOOR_ACTION_DIM 4
#else
#define OBS_SIZE SIXDOF_DOOR_OBS_DIM
#define NUM_ATNS 2
#define ACT_SIZES {1, 1}
#define FLIGHTRL_DOOR_ACTION_DIM 2
#endif
#define OBS_TENSOR_T FloatTensor

#include "native_door_env_types.inc"

static void c_reset(FlightRLDoorEnv *env), c_step(FlightRLDoorEnv *env);
static void c_render(FlightRLDoorEnv *env), c_close(FlightRLDoorEnv *env);

#define Env FlightRLDoorEnv
#include "vecenv.h"

#include "native_door_domain.inc"
#include "native_door_episode_groups.inc"
#include "native_door_lane.inc"

#include "native_door_env_config.inc"

static void c_reset(Env *env) {
    flightrl_door_episode_rng_next(&env->episode_rng, &env->rng, &env->appearance_rng);
    randomize_door_domain(env);
    float placeholder_target[3] = {0.0f, 0.0f, 0.8f};
    float placeholder_yaw = 0.0f;
    env->rng = flightrl_sixdof_reset_one(
        env->position,
        env->velocity,
        env->quaternion,
        env->body_rates,
        env->ranges,
        &env->thrust_state,
        env->physics,
        placeholder_target,
        &placeholder_yaw,
        env->low_level_action,
        &env->control_step,
        env->state_observation,
        env->rewards,
        &env->terminal,
        &env->truncation,
        env->room,
        env->rng
    );
    flightrl_door_scene_sample(
        env->position, env->quaternion, env->room, &env->scene, &env->rng,
        env->obstacle_probability, env->layout_diversity,
        env->mission.target_standoff_m,
        fminf(env->mission.planar_position_tolerance_m, env->mission.standoff_tolerance_m)
    );
    memset(env->velocity, 0, sizeof(env->velocity));
    memset(env->body_rates, 0, sizeof(env->body_rates));
    memset(env->previous_action, 0, sizeof(env->previous_action));
    memset(env->previous_frame, 0, sizeof(env->previous_frame));
    memcpy(env->origin_position, env->position, sizeof(env->origin_position));
    env->origin_yaw = flightrl_door_yaw(env->quaternion);
    env->takeoff_origin_z = env->room[4];
    update_ranges(env->position, env->quaternion, env->ranges, env->room);
    float camera_range = env->camera_mean_max - env->camera_mean_min;
    float normal_camera_min = env->camera_mean_min + 0.25f * camera_range;
    int low_light = (
        env->camera_randomization > 0.5f
        && rnd(&env->appearance_rng, 0.0f, 1.0f) < 0.10f
    );
    env->camera_mean = low_light
        ? rnd(&env->appearance_rng, env->camera_mean_min, normal_camera_min)
        : rnd(&env->appearance_rng, normal_camera_min, env->camera_mean_max);
    env->scene.door[5] = (float)(
        flightrl_door_seed_mix(rng_next(&env->appearance_rng)) & 0x00ffffffu
    );
    env->scene_seed = (int)flightrl_door_seed_mix(rng_next(&env->appearance_rng));
    env->control_step = 0;
    flightrl_door_detector_reset(&env->detector);
    env->visible_steps = 0;
    env->mission_state.dwell_steps = 0;
    env->thrust_state = 1.0f;
    env->current_return = 0.0f;
    env->current_length = 0.0f;
    env->terminal = 0;
    env->truncation = 0;
    write_door_observation(env, 1);
    capture_door_episode_group(env, (uint8_t)low_light);
}

static void c_step(Env *env) {
    int previously_observed = env->scene.target_observed;
    float previous_distance = flightrl_door_scene_distance(
        env->position,
        &env->scene
    );
    float command[4];
    apply_door_lane_action(env, command);
    flightrl_sixdof_setpoint_actions_batch(
        env->velocity,
        env->quaternion,
        command,
        env->physics,
        env->low_level_action,
        1,
        env->max_horizontal_speed,
        env->max_vertical_speed,
        env->velocity_gain,
        env->attitude_gain,
        env->vertical_gain
    );
    float physics_dt = env->control_dt / fmaxf(
        (float)env->physics_substeps,
        1.0f
    );
    for (int i = 0; i < env->physics_substeps; ++i) {
        flightrl_sixdof_step_batch(
            env->position,
            env->velocity,
            env->quaternion,
            env->body_rates,
            env->ranges,
            &env->thrust_state,
            env->low_level_action,
            env->physics,
            env->room,
            1,
            physics_dt
        );
    }
    env->control_step += 1;
#ifdef FLIGHTRL_EDGE_STUDENT_LANE
    int visible = write_door_observation(env, 0);
#else
    int visible = flightrl_door_scene_visible(
        env->position,
        env->quaternion,
        env->room,
        &env->scene
    );
    env->scene.target_observed = env->scene.target_observed || visible;
    flightrl_door_teacher_advance(
        env->position,
        env->quaternion,
        &env->scene
    );
#endif
    env->visible_steps += visible;
    float distance = flightrl_door_scene_distance(env->position, &env->scene);
    int collision = flightrl_door_scene_collides(
        env->position,
        env->room,
        &env->scene
    );
    int success = flightrl_door_mission_step(
        &env->mission,
        &env->mission_state,
        env->position,
        env->velocity,
        env->quaternion,
        env->body_rates,
        env->room,
        (int)env->scene.door[0],
        env->scene.target,
        env->scene.target_yaw,
        visible
    );
    env->truncation = env->control_step >= env->max_episode_steps;
    env->terminal = collision || success;
    float action_cost = door_lane_action_cost(env);
    int discovered = !previously_observed && env->scene.target_observed;
    env->rewards[0] = (
        env->scene.target_observed
            ? 12.0f * (previous_distance - distance)
            : 0.0f
        )
        + 1.0f * discovered + 0.005f * visible - 0.0005f * action_cost
        - 10.0f * collision + 10.0f * success;
#ifndef FLIGHTRL_EDGE_STUDENT_LANE
    write_door_observation(env, 0);
#endif
    env->terminals[0] = (env->terminal || env->truncation) ? 1.0f : 0.0f;
    env->current_return += env->rewards[0];
    env->current_length += 1.0f;
    if (env->terminals[0]) {
        env->log.score += success;
        env->log.episode_return += env->current_return;
        env->log.episode_length += env->current_length;
        env->log.success_rate += success;
        env->log.collision_rate += collision;
        env->log.outside_fov_episode_fraction += env->scene.initial_outside_fov;
        env->log.outside_fov_success_fraction += (
            env->scene.initial_outside_fov && success
        );
        env->log.outside_fov_observed_fraction += (
            env->scene.initial_outside_fov && env->scene.target_observed
        );
        env->log.observed_episode_fraction += env->scene.target_observed;
        env->log.door_visible_fraction += (
            (float)env->visible_steps / fmaxf(env->current_length, 1.0f)
        );
        log_door_episode_group(env, (uint8_t)success);
        env->log.n += 1.0f;
        float terminal_reward = env->rewards[0];
        c_reset(env);
        env->rewards[0] = terminal_reward;
        env->terminals[0] = 1.0f;
    }
}

static void c_render(Env *env) { (void)env; }
static void c_close(Env *env) { (void)env; }
