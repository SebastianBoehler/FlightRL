from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from .puffer4_config import Puffer4ExportSettings, render_puffer4_ini
from .puffer4_vision_sections import build_visual_navigation_sections


VISUAL_NATIVE_FILES = (
    "native_sixdof.c",
    "native_sixdof.h",
    "native_sixdof_context.inc",
    "native_sixdof_step.inc",
    "native_sixdof_setpoint.c",
    "native_sixdof_setpoint.h",
    "native_sixdof_vision.c",
    "native_sixdof_vision.h",
)


@dataclass(slots=True)
class VisualPufferExportResult:
    env_name: str
    env_dir: Path
    config_path: Path


def render_visual_puffer4_binding() -> str:
    return r"""#include <math.h>
#include <stdint.h>
#include <string.h>

#include "native_sixdof.h"
#include "native_sixdof.c"
#include "native_sixdof_setpoint.h"
#include "native_sixdof_setpoint.c"
#include "native_sixdof_vision.h"
#include "native_sixdof_vision.c"

#define OBS_SIZE SIXDOF_VISION_OBS_DIM
#define NUM_ATNS 4
#define ACT_SIZES {1, 1, 1, 1}
#define OBS_TENSOR_T FloatTensor
#define PROGRESS_REWARD_SCALE 20.0f
#define ACTION_COST_SCALE 0.0005f
#define TERMINAL_REWARD 10.0f
#define AVOIDANCE_REWARD_SCALE 0.03f

typedef struct {
    float score;
    float episode_return;
    float episode_length;
    float success_rate;
    float collision_rate;
    float n;
} Log;

typedef struct {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    uint32_t rng;
    float control_dt;
    int physics_substeps;
    int max_episode_steps;
    float success_radius;
    float max_horizontal_speed;
    float max_vertical_speed;
    float velocity_gain;
    float attitude_gain;
    float vertical_gain;
    float camera_mean_min;
    float camera_mean_max;
    float obstacle_probability;
    float navigation_residual_scale;
    float waypoint_slowdown_distance;
    float camera_mean;
    int scene_seed;
    float room[7];
    float physics[SIXDOF_PHYSICS_DIM];
    float position[3];
    float velocity[3];
    float quaternion[4];
    float body_rates[3];
    float ranges[6];
    float thrust_state;
    float target_position[3];
    float target_yaw;
    float low_level_action[4];
    float state_observation[28];
    float obstacle[6];
    uint8_t previous_frame[SIXDOF_VISION_PIXELS];
    int control_step;
    int physics_step;
    unsigned char terminal;
    unsigned char truncation;
    float current_return;
    float current_length;
} FlightRLVisualEnv;

static void c_reset(FlightRLVisualEnv* env);
static void c_step(FlightRLVisualEnv* env);
static void c_render(FlightRLVisualEnv* env);
static void c_close(FlightRLVisualEnv* env);

#define Env FlightRLVisualEnv
#include "vecenv.h"

static float distance_to_target(Env* env) {
    float dx = env->target_position[0] - env->position[0];
    float dy = env->target_position[1] - env->position[1];
    float dz = env->target_position[2] - env->position[2];
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

static int collides(Env* env) {
    float margin = 0.06f;
    int room_hit = env->position[0] < env->room[0] + margin || env->position[0] > env->room[1] - margin ||
        env->position[1] < env->room[2] + margin || env->position[1] > env->room[3] - margin ||
        env->position[2] < env->room[4] + margin || env->position[2] > env->room[5] - margin;
    int obstacle_hit = env->position[0] > env->obstacle[0] - margin && env->position[0] < env->obstacle[1] + margin &&
        env->position[1] > env->obstacle[2] - margin && env->position[1] < env->obstacle[3] + margin &&
        env->position[2] > env->obstacle[4] - margin && env->position[2] < env->obstacle[5] + margin;
    return room_hit || obstacle_hit;
}

static void sample_scene(Env* env) {
    float side = rnd(&env->rng, 0.0f, 1.0f) < 0.5f ? -1.0f : 1.0f;
    env->position[0] = -1.25f * side;
    env->position[1] = rnd(&env->rng, -0.35f, 0.35f);
    env->position[2] = rnd(&env->rng, 0.50f, 0.75f);
    env->target_position[0] = 1.25f * side;
    env->target_position[1] = rnd(&env->rng, -0.35f, 0.35f);
    env->target_position[2] = env->position[2];
    float yaw = side > 0.0f ? 0.0f : 3.14159265f;
    env->quaternion[0] = cosf(0.5f * yaw);
    env->quaternion[1] = 0.0f;
    env->quaternion[2] = 0.0f;
    env->quaternion[3] = sinf(0.5f * yaw);
    env->target_yaw = yaw;
    float center_y = 0.5f * (env->position[1] + env->target_position[1]) + rnd(&env->rng, -0.30f, 0.30f);
    float half_x = rnd(&env->rng, 0.16f, 0.28f);
    float half_y = rnd(&env->rng, 0.35f, 0.55f);
    env->obstacle[0] = -half_x;
    env->obstacle[1] = half_x;
    env->obstacle[2] = center_y - half_y;
    env->obstacle[3] = center_y + half_y;
    env->obstacle[4] = 0.0f;
    env->obstacle[5] = rnd(&env->rng, 1.10f, 1.55f);
    if (rnd(&env->rng, 0.0f, 1.0f) >= env->obstacle_probability) {
        for (int i = 0; i < 3; ++i) {
            env->obstacle[2 * i] = 10.0f;
            env->obstacle[2 * i + 1] = 11.0f;
        }
    }
    update_ranges(env->position, env->quaternion, env->ranges, env->room);
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    uint32_t env_index = env->rng;
    env->rng = (uint32_t)dict_get(kwargs, "seed")->value +
        0x9e3779b9u * (env_index + 1u);
    env->control_dt = (float)dict_get(kwargs, "control_dt")->value;
    env->physics_substeps = (int)dict_get(kwargs, "physics_substeps")->value;
    env->max_episode_steps = (int)dict_get(kwargs, "max_episode_steps")->value;
    env->success_radius = (float)dict_get(kwargs, "success_radius_m")->value;
    env->max_horizontal_speed = (float)dict_get(kwargs, "max_horizontal_speed_m_s")->value;
    env->max_vertical_speed = (float)dict_get(kwargs, "max_vertical_speed_m_s")->value;
    env->velocity_gain = (float)dict_get(kwargs, "velocity_gain")->value;
    env->attitude_gain = (float)dict_get(kwargs, "attitude_gain")->value;
    env->vertical_gain = (float)dict_get(kwargs, "vertical_gain")->value;
    env->camera_mean_min = (float)dict_get(kwargs, "camera_mean_min")->value;
    env->camera_mean_max = (float)dict_get(kwargs, "camera_mean_max")->value;
    env->obstacle_probability = (float)dict_get(kwargs, "obstacle_probability")->value;
    env->navigation_residual_scale = (float)dict_get(kwargs, "navigation_residual_scale")->value;
    env->waypoint_slowdown_distance = (float)dict_get(kwargs, "waypoint_slowdown_distance_m")->value;
    const char* room_keys[7] = {"room_x_min", "room_x_max", "room_y_min", "room_y_max", "room_z_min", "room_z_max", "max_range_m"};
    const char* physics_keys[9] = {"mass_kg", "gravity_m_s2", "linear_drag", "rate_tau_s", "thrust_scale", "max_rate_roll", "max_rate_pitch", "max_rate_yaw", "motor_tau_s"};
    for (int i = 0; i < 7; ++i) env->room[i] = (float)dict_get(kwargs, room_keys[i])->value;
    for (int i = 0; i < 9; ++i) env->physics[i] = (float)dict_get(kwargs, physics_keys[i])->value;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "success_rate", log->success_rate);
    dict_set(out, "collision_rate", log->collision_rate);
}

static void write_visual_observation(Env* env, uint8_t reset_temporal) {
    flightrl_sixdof_visual_observation_scene(
        env->position, env->quaternion, env->target_position, env->target_yaw, env->room, env->obstacle,
        env->camera_mean, env->scene_seed, env->previous_frame, reset_temporal, env->observations);
}

static void c_reset(Env* env) {
    env->rng = flightrl_sixdof_reset_one(
        env->position, env->velocity, env->quaternion, env->body_rates, env->ranges, &env->thrust_state,
        env->physics, env->target_position, &env->target_yaw, env->low_level_action, &env->physics_step,
        env->state_observation, env->rewards, &env->terminal, &env->truncation, env->room, env->rng);
    sample_scene(env);
    env->camera_mean = rnd(&env->rng, env->camera_mean_min, env->camera_mean_max);
    env->scene_seed = (int)rng_next(&env->rng);
    memset(env->previous_frame, 0, sizeof(env->previous_frame));
    env->control_step = 0;
    env->physics_step = 0;
    env->thrust_state = 1.0f;
    env->current_return = 0.0f;
    env->current_length = 0.0f;
    env->terminal = 0;
    env->truncation = 0;
    write_visual_observation(env, 1);
}

static void c_step(Env* env) {
    float residual[4];
    for (int i = 0; i < 4; ++i) residual[i] = clampf(env->actions[i], -1.0f, 1.0f);
    float previous_distance = distance_to_target(env);
    flightrl_sixdof_waypoint_residual_actions_batch(
        env->position, env->velocity, env->quaternion, env->target_position, &env->target_yaw,
        residual, env->physics, env->low_level_action, 1, env->max_horizontal_speed,
        env->max_vertical_speed, env->velocity_gain, env->attitude_gain, env->vertical_gain,
        env->navigation_residual_scale, env->waypoint_slowdown_distance);
    float physics_dt = env->control_dt / fmaxf((float)env->physics_substeps, 1.0f);
    for (int i = 0; i < env->physics_substeps; ++i) {
        flightrl_sixdof_step_batch(
            env->position, env->velocity, env->quaternion, env->body_rates, env->ranges, &env->thrust_state,
            env->low_level_action, env->physics, env->room, 1, physics_dt);
    }
    env->control_step += 1;
    float distance = distance_to_target(env);
    int collision = collides(env);
    int success = distance <= env->success_radius;
    env->truncation = env->control_step >= env->max_episode_steps;
    env->terminal = collision || success;
    float action_cost = residual[0]*residual[0] + residual[1]*residual[1] +
        residual[2]*residual[2] + residual[3]*residual[3];
    float avoidance = flightrl_sixdof_avoidance_alignment(env->position, env->quaternion, env->obstacle, residual);
    env->rewards[0] = PROGRESS_REWARD_SCALE*(previous_distance - distance) -
        ACTION_COST_SCALE*action_cost + AVOIDANCE_REWARD_SCALE*avoidance -
        TERMINAL_REWARD*collision + TERMINAL_REWARD*success;
    env->scene_seed += 1;
    write_visual_observation(env, 0);
    env->terminals[0] = (env->terminal || env->truncation) ? 1.0f : 0.0f;
    env->current_return += env->rewards[0];
    env->current_length += 1.0f;
    if (env->terminals[0]) {
        env->log.score += success;
        env->log.episode_return += env->current_return;
        env->log.episode_length += env->current_length;
        env->log.success_rate += success;
        env->log.collision_rate += collision;
        env->log.n += 1.0f;
        float terminal_reward = env->rewards[0];
        c_reset(env);
        env->rewards[0] = terminal_reward;
        env->terminals[0] = 1.0f;
    }
}

static void c_render(Env* env) { (void)env; }
static void c_close(Env* env) { (void)env; }
"""


def export_visual_puffer4_assets(
    pufferlib_root: str | Path,
    settings: Puffer4ExportSettings | None = None,
) -> VisualPufferExportResult:
    resolved = settings or Puffer4ExportSettings(
        env_name="flightrl_visual_navigation",
        total_agents=128,
        num_buffers=1,
        num_threads=8,
        policy_hidden_size=128,
        policy_num_layers=1,
    )
    root = Path(pufferlib_root).expanduser().resolve()
    env_dir = root / "ocean" / resolved.env_name
    config_path = root / "config" / f"{resolved.env_name}.ini"
    env_dir.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    native_dir = Path(__file__).resolve().parent / "native"
    for filename in VISUAL_NATIVE_FILES:
        shutil.copy2(native_dir / filename, env_dir / filename)
    (env_dir / "binding.c").write_text(render_visual_puffer4_binding())
    config_path.write_text(render_puffer4_ini(build_visual_navigation_sections(resolved)))
    return VisualPufferExportResult(resolved.env_name, env_dir, config_path)
