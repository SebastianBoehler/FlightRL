from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from .binding_kwargs import build_binding_kwargs
from .config import FlightConfig
from .puffer4_config import Puffer4ExportSettings, build_puffer4_sections, render_puffer4_ini


PUFFER4_NATIVE_FILES = (
    "native_actions.c",
    "native_dynamics.c",
    "native_env.c",
    "native_env.h",
    "native_logging.c",
    "native_observation.c",
    "native_reset.c",
    "native_reward.c",
    "native_rng.h",
    "native_tasks.c",
    "native_termination.c",
    "native_types.h",
    "native_wind.c",
)

FIELD_ASSIGNMENTS = (
    ("dt", "env->inner.dt", "float"),
    ("action_dim", "env->inner.sensor_config.action_dim", "int"),
    ("observation_dim", "env->inner.sensor_config.observation_dim", "int"),
    ("observation_flags", "env->inner.sensor_config.flags", "int"),
    ("state_noise_std", "env->inner.sensor_config.state_noise_std", "float"),
    ("imu_noise_std", "env->inner.sensor_config.imu_noise_std", "float"),
    ("task_type", "env->inner.task_config.task_type", "int"),
    ("action_mode", "env->inner.task_config.action_mode", "int"),
    ("reset_mode", "env->inner.task_config.reset_mode", "int"),
    ("max_steps", "env->inner.task_config.max_steps", "int"),
    ("sequence_length", "env->inner.task_config.sequence_length", "int"),
    ("hover_hold_steps", "env->inner.task_config.hover_hold_steps", "int"),
    ("success_radius", "env->inner.task_config.success_radius", "float"),
    ("hover_speed_threshold", "env->inner.task_config.hover_speed_threshold", "float"),
    ("fixed_start_x", "env->inner.task_config.fixed_start_x", "float"),
    ("fixed_start_z", "env->inner.task_config.fixed_start_z", "float"),
    ("fixed_target_x", "env->inner.task_config.fixed_target_x", "float"),
    ("fixed_target_z", "env->inner.task_config.fixed_target_z", "float"),
    ("spawn_x_min", "env->inner.task_config.spawn_x_min", "float"),
    ("spawn_x_max", "env->inner.task_config.spawn_x_max", "float"),
    ("spawn_z_min", "env->inner.task_config.spawn_z_min", "float"),
    ("spawn_z_max", "env->inner.task_config.spawn_z_max", "float"),
    ("target_x_min", "env->inner.task_config.target_x_min", "float"),
    ("target_x_max", "env->inner.task_config.target_x_max", "float"),
    ("target_z_min", "env->inner.task_config.target_z_min", "float"),
    ("target_z_max", "env->inner.task_config.target_z_max", "float"),
    ("mass", "env->inner.drone_config.mass", "float"),
    ("inertia", "env->inner.drone_config.inertia", "float"),
    ("arm_length", "env->inner.drone_config.arm_length", "float"),
    ("drag", "env->inner.drone_config.drag", "float"),
    ("angular_drag", "env->inner.drone_config.angular_drag", "float"),
    ("gravity", "env->inner.drone_config.gravity", "float"),
    ("hover_thrust", "env->inner.drone_config.hover_thrust", "float"),
    ("thrust_gain", "env->inner.drone_config.thrust_gain", "float"),
    ("max_total_thrust", "env->inner.drone_config.max_total_thrust", "float"),
    ("max_pitch_torque", "env->inner.drone_config.max_pitch_torque", "float"),
    ("actuator_tau", "env->inner.drone_config.actuator_tau", "float"),
    ("max_velocity", "env->inner.drone_config.max_velocity", "float"),
    ("max_pitch_rate", "env->inner.drone_config.max_pitch_rate", "float"),
    ("max_pitch_angle", "env->inner.drone_config.max_pitch_angle", "float"),
    ("floor_z", "env->inner.drone_config.floor_z", "float"),
    ("x_limit", "env->inner.drone_config.x_limit", "float"),
    ("z_limit", "env->inner.drone_config.z_limit", "float"),
    ("alive_bonus", "env->inner.reward_config.alive_bonus", "float"),
    ("distance_penalty", "env->inner.reward_config.distance_penalty", "float"),
    ("progress_bonus", "env->inner.reward_config.progress_bonus", "float"),
    ("velocity_penalty", "env->inner.reward_config.velocity_penalty", "float"),
    ("angular_rate_penalty", "env->inner.reward_config.angular_rate_penalty", "float"),
    ("control_penalty", "env->inner.reward_config.control_penalty", "float"),
    ("smoothness_penalty", "env->inner.reward_config.smoothness_penalty", "float"),
    ("success_bonus", "env->inner.reward_config.success_bonus", "float"),
    ("crash_penalty", "env->inner.reward_config.crash_penalty", "float"),
    ("out_of_bounds_penalty", "env->inner.reward_config.out_of_bounds_penalty", "float"),
    ("randomization_enabled", "env->inner.randomization_config.enabled", "int"),
    ("mass_scale", "env->inner.randomization_config.mass_scale", "float"),
    ("drag_scale", "env->inner.randomization_config.drag_scale", "float"),
    ("thrust_scale", "env->inner.randomization_config.thrust_scale", "float"),
    ("actuator_tau_scale", "env->inner.randomization_config.actuator_tau_scale", "float"),
    ("sensor_noise_scale", "env->inner.randomization_config.sensor_noise_scale", "float"),
    ("wind_enabled", "env->inner.wind_config.enabled", "int"),
    ("wind_steady_x", "env->inner.wind_config.steady_x", "float"),
    ("wind_steady_z", "env->inner.wind_config.steady_z", "float"),
    ("wind_gust_strength", "env->inner.wind_config.gust_strength", "float"),
    ("wind_gust_tau", "env->inner.wind_config.gust_tau", "float"),
)


@dataclass(slots=True)
class Puffer4ExportResult:
    env_name: str
    env_dir: Path
    config_path: Path


def _generate_assignments() -> str:
    lines = []
    for key, target, ctype in FIELD_ASSIGNMENTS:
        cast = "int" if ctype == "int" else "float"
        lines.append(f'    {target} = ({cast})dict_get(kwargs, "{key}")->value;')
    return "\n".join(lines)


def _generate_waypoint_assignments() -> str:
    lines = []
    for idx in range(8):
        lines.append(
            f'    env->inner.task_config.fixed_waypoints[{idx}].x = (float)dict_get(kwargs, "waypoint_{idx}_x")->value;'
        )
        lines.append(
            f'    env->inner.task_config.fixed_waypoints[{idx}].z = (float)dict_get(kwargs, "waypoint_{idx}_z")->value;'
        )
    return "\n".join(lines)


def render_puffer4_binding(config: FlightConfig) -> str:
    action_sizes = ", ".join("1" for _ in range(config.action_dim))
    include_lines = "\n".join(f'#include "{name}"' for name in PUFFER4_NATIVE_FILES if name.endswith(".c"))
    return f"""#define c_reset flightrl_inner_reset
#define c_step flightrl_inner_step
#define c_close flightrl_inner_close
#include "native_env.h"
#undef c_reset
#undef c_step
#undef c_close

#define OBS_SIZE {config.observation_dim}
#define NUM_ATNS {config.action_dim}
#define ACT_SIZES {{{action_sizes}}}
#define OBS_TENSOR_T FloatTensor

typedef struct {{
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;
    unsigned char terminal_flag;
    unsigned char truncation_flag;
    DronePlanarEnv inner;
}} FlightRLPufferEnv;

static void c_reset(FlightRLPufferEnv* env);
static void c_step(FlightRLPufferEnv* env);
static void c_render(FlightRLPufferEnv* env);
static void c_close(FlightRLPufferEnv* env);

#define Env FlightRLPufferEnv
#include "vecenv.h"

#define c_reset flightrl_inner_reset
#define c_step flightrl_inner_step
#define c_close flightrl_inner_close
{include_lines}
#undef c_reset
#undef c_step
#undef c_close

static void flightrl_sync_inner(Env* env) {{
    env->inner.observations = env->observations;
    env->inner.actions = env->actions;
    env->inner.rewards = env->rewards;
    env->inner.terminals = &env->terminal_flag;
    env->inner.truncations = &env->truncation_flag;
    env->inner.log = env->log;
}}

static void flightrl_sync_outer(Env* env) {{
    env->log = env->inner.log;
    env->terminals[0] = (env->terminal_flag || env->truncation_flag) ? 1.0f : 0.0f;
}}

void my_init(Env* env, Dict* kwargs) {{
    env->num_agents = 1;
{_generate_assignments()}
{_generate_waypoint_assignments()}
    if (env->inner.sensor_config.flags & (FLIGHT_OBS_RANGE | FLIGHT_OBS_VISION)) {{
        fprintf(stderr, "FlightRL Puffer4 export does not support range or vision sensors\\n");
        abort();
    }}
    env->terminal_flag = 0;
    env->truncation_flag = 0;
    env->inner.rng_state = (uint64_t)dict_get(kwargs, "seed")->value + 0x9e3779b97f4a7c15ULL;
}}

void my_log(Log* log, Dict* out) {{
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "success_rate", log->success_rate);
    dict_set(out, "crash_rate", log->crash_rate);
    dict_set(out, "timeout_rate", log->timeout_rate);
    dict_set(out, "mean_distance", log->mean_distance);
    dict_set(out, "mean_action_magnitude", log->mean_action_magnitude);
}}

static void c_reset(Env* env) {{
    flightrl_sync_inner(env);
    flightrl_inner_reset(&env->inner);
    flightrl_sync_outer(env);
}}

static void c_step(Env* env) {{
    flightrl_sync_inner(env);
    flightrl_inner_step(&env->inner);
    flightrl_sync_outer(env);
}}

static void c_render(Env* env) {{
    (void)env;
}}

static void c_close(Env* env) {{
    flightrl_sync_inner(env);
    flightrl_inner_close(&env->inner);
}}
"""


def export_puffer4_assets(
    config: FlightConfig,
    pufferlib_root: str | Path,
    settings: Puffer4ExportSettings | None = None,
) -> Puffer4ExportResult:
    resolved_settings = settings or Puffer4ExportSettings()
    root = Path(pufferlib_root).expanduser().resolve()
    env_dir = root / "ocean" / resolved_settings.env_name
    config_path = root / "config" / f"{resolved_settings.env_name}.ini"
    env_dir.mkdir(parents=True, exist_ok=True)

    native_dir = Path(__file__).resolve().parent / "native"
    for filename in PUFFER4_NATIVE_FILES:
        shutil.copy2(native_dir / filename, env_dir / filename)

    binding_source = render_puffer4_binding(config)
    (env_dir / "binding.c").write_text(binding_source)

    env_values = build_binding_kwargs(config)
    sections = build_puffer4_sections(config, env_values, resolved_settings)
    config_path.write_text(render_puffer4_ini(sections))
    return Puffer4ExportResult(
        env_name=resolved_settings.env_name,
        env_dir=env_dir,
        config_path=config_path,
    )
