from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from .puffer4_config import Puffer4ExportSettings, render_puffer4_ini


SIXDOF_NATIVE_FILES = ("native_sixdof.c", "native_sixdof.h", "native_sixdof_context.inc")


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
    int step_count;
    unsigned char terminal;
    unsigned char truncation;
} FlightRLSixDofEnv;

static void c_reset(FlightRLSixDofEnv* env);
static void c_step(FlightRLSixDofEnv* env);
static void c_render(FlightRLSixDofEnv* env);
static void c_close(FlightRLSixDofEnv* env);

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
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "reward", log->reward);
}

static void c_reset(Env* env) {
    env->rng = flightrl_sixdof_reset_one(env->position, env->velocity, env->quaternion, env->body_rates, env->ranges,
        env->target_position, &env->target_yaw, env->previous_action, &env->step_count,
        env->observations, env->rewards, &env->terminal, &env->truncation, env->room, env->rng);
}

static void c_step(Env* env) {
    flightrl_sixdof_step_env_batch(env->position, env->velocity, env->quaternion, env->body_rates, env->ranges,
        env->target_position, &env->target_yaw, env->previous_action, &env->step_count, env->actions,
        env->observations, env->rewards, &env->terminal, &env->truncation, env->room, 1, env->dt);
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


def build_sixdof_sections(settings: Puffer4ExportSettings) -> dict[str, dict[str, int | float | str]]:
    total_agents = settings.total_agents or 4096
    num_buffers = settings.num_buffers or 8
    hidden_size = settings.policy_hidden_size or 128
    return {
        "base": {"env_name": settings.env_name, "checkpoint_interval": 10, "seed": settings.train_seed},
        "vec": {"total_agents": total_agents, "num_buffers": num_buffers, "num_threads": settings.num_threads or num_buffers},
        "env": {
            "seed": settings.train_seed,
            "dt": 0.01,
            "room_x_min": -2.0,
            "room_x_max": 2.0,
            "room_y_min": -2.0,
            "room_y_max": 2.0,
            "room_z_min": 0.0,
            "room_z_max": 2.5,
            "max_range_m": 4.0,
        },
        "policy": {"hidden_size": hidden_size, "num_layers": settings.policy_num_layers, "expansion_factor": 1},
        "torch": {"network": "MLP", "encoder": "DefaultEncoder", "decoder": "DefaultDecoder"},
        "train": {
            "gpus": 1,
            "seed": settings.train_seed,
            "total_timesteps": 1048576,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "replay_ratio": 2,
            "clip_coef": 0.2,
            "vf_coef": 0.5,
            "ent_coef": 0.001,
            "minibatch_size": 8192,
            "horizon": 32,
        },
    }
