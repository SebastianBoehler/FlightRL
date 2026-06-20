from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from .puffer4_config import Puffer4ExportSettings, render_puffer4_ini
from .puffer4_sixdof_sections import build_sixdof_sections


SIXDOF_NATIVE_FILES = ("native_sixdof.c", "native_sixdof.h", "native_sixdof_context.inc", "native_sixdof_step.inc")


@dataclass(slots=True)
class SixDofPufferExportResult:
    env_name: str
    env_dir: Path
    config_path: Path


def render_sixdof_puffer4_binding() -> str:
    return """#include <math.h>
#include <stdint.h>
#include <string.h>

#include "native_sixdof.h"
#include "native_sixdof.c"

#define OBS_SIZE 28
#define NUM_ATNS 4
#define ACT_SIZES {1, 1, 1, 1}
#define OBS_TENSOR_T FloatTensor

typedef struct {
    float episode_return;
    float episode_length;
    float reward;
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
    float dt;
    float position[3];
    float velocity[3];
    float quaternion[4];
    float body_rates[3];
    float ranges[6];
	    float room[7];
	    float target_position[3];
	    float target_yaw;
	    float previous_action[4];
	    float command_state[4];
	    float thrust_state;
	    float physics[SIXDOF_PHYSICS_DIM];
	    float state_noise_std_m;
	    float velocity_noise_std_m_s;
	    float body_rate_noise_std_rad_s;
	    float range_noise_std_m;
	    float range_dropout_prob;
	    float action_lag_s;
	    int task_id;
	    int reward_mode;
	    float near_wall_probability;
	    float near_wall_min_clearance_m;
	    float near_wall_max_clearance_m;
	    float near_wall_yaw_jitter_rad;
	    float reset_z_min;
	    float reset_z_max;
	    float target_z_min;
	    float target_z_max;
	    float target_xy_offset_abs;
	    float target_z_offset_abs;
	    float target_yaw_offset_abs;
	    int step_count;
	    unsigned char terminal;
	    unsigned char truncation;
	} FlightRLSixDofEnv;

	static void c_reset(FlightRLSixDofEnv* env);
	static void c_step(FlightRLSixDofEnv* env);
	static void c_render(FlightRLSixDofEnv* env);
	static void c_close(FlightRLSixDofEnv* env);
	static void apply_observation_profile(FlightRLSixDofEnv* env);
	static void apply_reset_profile(FlightRLSixDofEnv* env);

#define Env FlightRLSixDofEnv
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->dt = (float)dict_get(kwargs, "dt")->value;
    env->rng = (uint32_t)dict_get(kwargs, "seed")->value + 0x9e3779b9u;
    env->room[0] = (float)dict_get(kwargs, "room_x_min")->value;
    env->room[1] = (float)dict_get(kwargs, "room_x_max")->value;
    env->room[2] = (float)dict_get(kwargs, "room_y_min")->value;
    env->room[3] = (float)dict_get(kwargs, "room_y_max")->value;
    env->room[4] = (float)dict_get(kwargs, "room_z_min")->value;
    env->room[5] = (float)dict_get(kwargs, "room_z_max")->value;
    env->room[6] = (float)dict_get(kwargs, "max_range_m")->value;
    env->physics[0] = (float)dict_get(kwargs, "mass_kg")->value;
    env->physics[1] = (float)dict_get(kwargs, "gravity_m_s2")->value;
    env->physics[2] = (float)dict_get(kwargs, "linear_drag")->value;
    env->physics[3] = (float)dict_get(kwargs, "rate_tau_s")->value;
    env->physics[4] = (float)dict_get(kwargs, "thrust_scale")->value;
    env->physics[5] = (float)dict_get(kwargs, "max_rate_roll")->value;
	    env->physics[6] = (float)dict_get(kwargs, "max_rate_pitch")->value;
	    env->physics[7] = (float)dict_get(kwargs, "max_rate_yaw")->value;
	    env->physics[8] = (float)dict_get(kwargs, "motor_tau_s")->value;
	    env->state_noise_std_m = (float)dict_get(kwargs, "state_noise_std_m")->value;
	    env->velocity_noise_std_m_s = (float)dict_get(kwargs, "velocity_noise_std_m_s")->value;
	    env->body_rate_noise_std_rad_s = (float)dict_get(kwargs, "body_rate_noise_std_rad_s")->value;
	    env->range_noise_std_m = (float)dict_get(kwargs, "range_noise_std_m")->value;
	    env->range_dropout_prob = (float)dict_get(kwargs, "range_dropout_prob")->value;
	    env->action_lag_s = (float)dict_get(kwargs, "action_lag_s")->value;
	    env->task_id = (int)dict_get(kwargs, "task_id")->value;
	    env->reward_mode = (int)dict_get(kwargs, "reward_mode")->value;
	    env->near_wall_probability = (float)dict_get(kwargs, "near_wall_probability")->value;
	    env->near_wall_min_clearance_m = (float)dict_get(kwargs, "near_wall_min_clearance_m")->value;
	    env->near_wall_max_clearance_m = (float)dict_get(kwargs, "near_wall_max_clearance_m")->value;
	    env->near_wall_yaw_jitter_rad = (float)dict_get(kwargs, "near_wall_yaw_jitter_rad")->value;
	    env->reset_z_min = (float)dict_get(kwargs, "reset_z_min")->value;
	    env->reset_z_max = (float)dict_get(kwargs, "reset_z_max")->value;
	    env->target_z_min = (float)dict_get(kwargs, "target_z_min")->value;
	    env->target_z_max = (float)dict_get(kwargs, "target_z_max")->value;
	    env->target_xy_offset_abs = (float)dict_get(kwargs, "target_xy_offset_abs")->value;
	    env->target_z_offset_abs = (float)dict_get(kwargs, "target_z_offset_abs")->value;
	    env->target_yaw_offset_abs = (float)dict_get(kwargs, "target_yaw_offset_abs")->value;
	}

void my_log(Log* log, Dict* out) {
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "reward", log->reward);
}

	static void c_reset(Env* env) {
	    for (int i = 0; i < 4; ++i) {
	        env->command_state[i] = 0.0f;
	    }
	    env->rng = flightrl_sixdof_reset_one(env->position, env->velocity, env->quaternion, env->body_rates, env->ranges,
	        &env->thrust_state, env->physics,
	        env->target_position, &env->target_yaw, env->previous_action, &env->step_count,
	        env->observations, env->rewards, &env->terminal, &env->truncation, env->room, env->rng);
	    apply_reset_profile(env);
	    float previous_error = task_position_error(env->position, env->target_position, env->task_id);
	    assemble_one(env->position, env->velocity, env->quaternion, env->body_rates, env->ranges,
	        env->target_position, env->target_yaw, env->previous_action, 0, env->task_id, env->reward_mode, previous_error,
	        env->physics, env->observations, env->rewards, &env->terminal, &env->truncation, env->room);
	    env->rewards[0] = 0.0f;
	    env->terminal = 0;
	    env->truncation = 0;
	    apply_observation_profile(env);
	}

	static void c_step(Env* env) {
	    float alpha = env->action_lag_s > 0.0f ? env->dt / (env->action_lag_s + env->dt) : 1.0f;
	    for (int i = 0; i < 4; ++i) {
	        float target = clampf(env->actions[i], -1.0f, 1.0f);
	        env->command_state[i] += alpha * (target - env->command_state[i]);
	    }
	    int task_ids[1] = {env->task_id};
	    float previous_error[1] = {task_position_error(env->position, env->target_position, env->task_id)};
	    flightrl_sixdof_step_env_context_batch(env->position, env->velocity, env->quaternion, env->body_rates, env->ranges,
	        &env->thrust_state, env->physics, env->target_position, &env->target_yaw, env->previous_action, &env->step_count, env->command_state,
	        env->observations, env->rewards, &env->terminal, &env->truncation, env->room, task_ids, env->reward_mode, previous_error, 1, env->dt);
	    apply_observation_profile(env);
	    env->terminals[0] = (env->terminal || env->truncation) ? 1.0f : 0.0f;
    env->log.episode_return += env->rewards[0];
    env->log.episode_length += 1.0f;
    env->log.reward += env->rewards[0];
    if (env->terminals[0]) {
        env->log.n += 1.0f;
        c_reset(env);
    }
}

	static void c_render(Env* env) { (void)env; }
	static void c_close(Env* env) { (void)env; }

	static float uniform_noise(Env* env, float std) {
	    if (std <= 0.0f) {
	        return 0.0f;
	    }
	    return rnd(&env->rng, -1.7320508f * std, 1.7320508f * std);
	}

	static void apply_observation_profile(Env* env) {
	    float max_range = env->room[6] > 1.0e-6f ? env->room[6] : 4.0f;
	    env->observations[0] += uniform_noise(env, env->state_noise_std_m) / 2.0f;
	    env->observations[1] += uniform_noise(env, env->state_noise_std_m) / 2.0f;
	    env->observations[2] += uniform_noise(env, env->state_noise_std_m) / 1.5f;
	    for (int i = 0; i < 3; ++i) {
	        env->observations[3 + i] += uniform_noise(env, env->velocity_noise_std_m_s) / 3.0f;
	        env->observations[10 + i] += uniform_noise(env, env->body_rate_noise_std_rad_s) / fmaxf(env->physics[5 + i], 1.0e-6f);
	    }
	    for (int i = 0; i < 6; ++i) {
	        if (env->range_dropout_prob > 0.0f && rnd(&env->rng, 0.0f, 1.0f) < env->range_dropout_prob) {
	            env->observations[18 + i] = 1.0f;
	        } else {
	            env->observations[18 + i] = clampf(env->observations[18 + i] + uniform_noise(env, env->range_noise_std_m) / max_range, 0.0f, 1.0f);
	        }
	    }
	}

	static void apply_reset_profile(Env* env) {
	    if (env->near_wall_probability > 0.0f && rnd(&env->rng, 0.0f, 1.0f) < env->near_wall_probability) {
	        float clearance = rnd(&env->rng, env->near_wall_min_clearance_m, env->near_wall_max_clearance_m);
	        int side = (int)rnd(&env->rng, 0.0f, 3.999f);
	        if (side == 0) env->position[0] = env->room[1] - clearance;
	        else if (side == 1) env->position[0] = env->room[0] + clearance;
	        else if (side == 2) env->position[1] = env->room[3] - clearance;
	        else env->position[1] = env->room[2] + clearance;
	        float yaw = rnd(&env->rng, -env->near_wall_yaw_jitter_rad, env->near_wall_yaw_jitter_rad);
	        env->quaternion[0] = cosf(0.5f * yaw);
	        env->quaternion[1] = 0.0f;
	        env->quaternion[2] = 0.0f;
	        env->quaternion[3] = sinf(0.5f * yaw);
	    }
	    env->position[2] = clampf(rnd(&env->rng, env->reset_z_min, env->reset_z_max), env->room[4] + 0.12f, env->room[5] - 0.12f);
	    if (env->target_xy_offset_abs >= 0.0f) {
	        env->target_position[0] = clampf(env->position[0] + rnd(&env->rng, -env->target_xy_offset_abs, env->target_xy_offset_abs), env->room[0] + 0.25f, env->room[1] - 0.25f);
	        env->target_position[1] = clampf(env->position[1] + rnd(&env->rng, -env->target_xy_offset_abs, env->target_xy_offset_abs), env->room[2] + 0.25f, env->room[3] - 0.25f);
	    }
	    if (env->target_z_offset_abs >= 0.0f) {
	        env->target_position[2] = clampf(env->position[2] + rnd(&env->rng, -env->target_z_offset_abs, env->target_z_offset_abs), env->room[4] + 0.35f, env->room[5] - 0.25f);
	    } else {
	        env->target_position[2] = clampf(rnd(&env->rng, env->target_z_min, env->target_z_max), env->room[4] + 0.35f, env->room[5] - 0.25f);
	    }
	    if (env->target_yaw_offset_abs >= 0.0f) {
	        env->target_yaw = wrap_angle(yaw_from_quat(env->quaternion) + rnd(&env->rng, -env->target_yaw_offset_abs, env->target_yaw_offset_abs));
	    }
	    update_ranges(env->position, env->quaternion, env->ranges, env->room);
	}
	"""


def export_sixdof_puffer4_assets(
    pufferlib_root: str | Path,
    settings: Puffer4ExportSettings | None = None,
) -> SixDofPufferExportResult:
    resolved = settings or Puffer4ExportSettings(env_name="flightrl_sixdof")
    root = Path(pufferlib_root).expanduser().resolve()
    env_dir = root / "ocean" / resolved.env_name
    config_path = root / "config" / f"{resolved.env_name}.ini"
    env_dir.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    native_dir = Path(__file__).resolve().parent / "native"
    for filename in SIXDOF_NATIVE_FILES:
        shutil.copy2(native_dir / filename, env_dir / filename)
    (env_dir / "binding.c").write_text(render_sixdof_puffer4_binding())
    config_path.write_text(render_puffer4_ini(build_sixdof_sections(resolved)))
    return SixDofPufferExportResult(resolved.env_name, env_dir, config_path)
