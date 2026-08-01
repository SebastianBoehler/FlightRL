from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from flightrl import load_config
from flightrl.binding_kwargs import build_binding_kwargs
from flightrl.puffer4_config import Puffer4ExportSettings, build_puffer4_sections
from flightrl.puffer4_export import export_puffer4_assets
from flightrl.puffer4_sixdof_export import export_sixdof_puffer4_assets
from flightrl.puffer4_sixdof_sections import build_sixdof_sections
from flightrl.sixdof import AxisAlignedObstacle, BoxRoom, SixDofCrazyflieEnv


ROOT = Path(__file__).resolve().parents[1]
VECENV_STUB = """
#pragma once
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { float* data; } FloatTensor;
typedef struct { const char* key; double value; void* ptr; } DictItem;
typedef struct { DictItem* items; int size; int capacity; } Dict;
static DictItem* dict_get(Dict* dict, const char* key) {
    for (int i = 0; i < dict->size; ++i) {
        if (strcmp(dict->items[i].key, key) == 0) return &dict->items[i];
    }
    abort();
}
static void dict_set(Dict* dict, const char* key, double value) {
    (void)dict; (void)key; (void)value;
}
"""


def _items(values: dict[str, int | float]) -> str:
    rows = [f'{{"{key}", {float(value):.17g}, NULL}}' for key, value in values.items()]
    return ",\n        ".join(rows)


def _compile_binding(
    tmp_path: Path,
    env_dir: Path,
    env_values: dict[str, int | float],
    harness_body: str,
    name: str,
) -> ctypes.CDLL:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the Puffer binding contract tests")
    (env_dir / "vecenv.h").write_text(VECENV_STUB)
    harness = tmp_path / f"{name}.c"
    harness.write_text(
        '#include "binding.c"\n'
        "static Dict kwargs(void) {\n"
        f"    static DictItem items[] = {{{_items(env_values)}}};\n"
        "    Dict out = {items, (int)(sizeof(items) / sizeof(items[0])), (int)(sizeof(items) / sizeof(items[0]))};\n"
        "    return out;\n"
        "}\n"
        + harness_body
    )
    library_path = tmp_path / f"{name}.so"
    subprocess.run(
        (compiler, "-std=c11", "-shared", "-fPIC", "-I", str(env_dir), str(harness), "-lm", "-o", str(library_path)),
        check=True,
        capture_output=True,
        text=True,
    )
    return ctypes.CDLL(str(library_path))


def test_compiled_planar_puffer_envs_get_independent_rng_and_keep_terminal_reward(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/tasks/hover.toml")
    settings = Puffer4ExportSettings(train_seed=31)
    result = export_puffer4_assets(config, tmp_path / "puffer", settings=settings)
    values = build_puffer4_sections(config, build_binding_kwargs(config), settings)["env"]
    library = _compile_binding(
        tmp_path,
        result.env_dir,
        values,
        """
uint64_t seed_for(uint32_t index) {
    Env env = {0}; env.rng = index; Dict args = kwargs(); my_init(&env, &args); return env.inner.rng_state;
}
void terminal_transition(uint32_t index, float* out) {
    float obs[OBS_SIZE] = {0}, actions[NUM_ATNS] = {0}, reward = 0.0f, terminal = 0.0f;
    Env env = {0}; env.rng = index; env.observations = obs; env.actions = actions; env.rewards = &reward; env.terminals = &terminal;
    Dict args = kwargs(); my_init(&env, &args); c_reset(&env);
    env.inner.drone.z = env.inner.runtime_dynamics.floor_z - 1.0f; c_step(&env);
    out[0] = reward; out[1] = terminal; out[2] = env.inner.drone.z;
}
""",
        "planar_contract",
    )
    library.seed_for.argtypes = (ctypes.c_uint32,)
    library.seed_for.restype = ctypes.c_uint64
    seeds = [library.seed_for(index) for index in range(4096)]
    assert len(set(seeds)) == len(seeds)
    assert seeds == [library.seed_for(index) for index in range(4096)]

    library.terminal_transition.argtypes = (ctypes.c_uint32, ctypes.POINTER(ctypes.c_float))
    transition = (ctypes.c_float * 3)()
    library.terminal_transition(0, transition)
    assert transition[1] == 1.0
    assert transition[0] != 0.0
    assert transition[2] > config.drone.floor_z


def test_compiled_sixdof_puffer_reset_matches_drift_profile_and_terminal_contract(tmp_path: Path) -> None:
    settings = Puffer4ExportSettings(train_seed=43, reset_profile="obstacle_hover_drift_recovery")
    result = export_sixdof_puffer4_assets(tmp_path / "puffer", settings=settings)
    values = build_sixdof_sections(settings)["env"]
    library = _compile_binding(
        tmp_path,
        result.env_dir,
        values,
        """
uint32_t seed_for(uint32_t index) {
    Env env = {0}; env.rng = index; Dict args = kwargs(); my_init(&env, &args); return env.rng;
}
void sample_reset(uint32_t index, float* out) {
    float obs[OBS_SIZE] = {0}, actions[NUM_ATNS] = {0}, reward = 0.0f, terminal = 0.0f;
    Env env = {0}; env.rng = index; env.observations = obs; env.actions = actions; env.rewards = &reward; env.terminals = &terminal;
    Dict args = kwargs(); my_init(&env, &args); c_reset(&env);
    memcpy(out, env.position, 3 * sizeof(float)); memcpy(out + 3, env.velocity, 3 * sizeof(float));
    memcpy(out + 6, env.quaternion, 4 * sizeof(float)); memcpy(out + 10, env.target_position, 3 * sizeof(float));
}
void terminal_transition(uint32_t index, float* out) {
    float obs[OBS_SIZE] = {0}, actions[NUM_ATNS] = {0}, reward = 0.0f, terminal = 0.0f;
    Env env = {0}; env.rng = index; env.observations = obs; env.actions = actions; env.rewards = &reward; env.terminals = &terminal;
    Dict args = kwargs(); my_init(&env, &args); c_reset(&env);
    env.position[0] = env.room[1] - 0.031f; env.velocity[0] = 10.0f; c_step(&env);
    out[0] = reward; out[1] = terminal; out[2] = env.position[0];
}
""",
        "sixdof_contract",
    )
    library.seed_for.argtypes = (ctypes.c_uint32,)
    library.seed_for.restype = ctypes.c_uint32
    seeds = [library.seed_for(index) for index in range(4096)]
    assert len(set(seeds)) == len(seeds)
    library.sample_reset.argtypes = (ctypes.c_uint32, ctypes.POINTER(ctypes.c_float))
    samples = np.empty((512, 13), dtype=np.float32)
    for index, row in enumerate(samples):
        library.sample_reset(index, row.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    assert len(np.unique(samples[:, :3], axis=0)) == len(samples)
    assert np.all(np.abs(samples[:, :2]) <= 0.55)
    assert np.all((samples[:, 2] >= 0.44) & (samples[:, 2] <= 0.58))
    np.testing.assert_allclose(samples[:, 10:13], samples[:, :3], atol=1e-7)
    assert np.std(samples[:, 3:5]) > 0.2
    assert np.std(samples[:, 5]) > 0.03
    assert np.max(np.abs(samples[:, 7:9])) > 0.01

    library.terminal_transition.argtypes = (ctypes.c_uint32, ctypes.POINTER(ctypes.c_float))
    transition = (ctypes.c_float * 3)()
    library.terminal_transition(0, transition)
    assert transition[1] == 1.0
    assert transition[0] != 0.0
    assert -2.0 < transition[2] < 2.0


def test_native_sixdof_rejects_interior_obstacles() -> None:
    room = BoxRoom(obstacles=(AxisAlignedObstacle(-0.2, 0.2, -0.2, 0.2, 0.0, 1.0),))
    with pytest.raises(ValueError, match="does not support BoxRoom interior obstacles"):
        SixDofCrazyflieEnv(num_envs=1, room=room, use_native_step=True)

    python_env = SixDofCrazyflieEnv(num_envs=1, room=room, use_native_step=False)
    assert python_env.room.obstacles == room.obstacles

    native_env = SixDofCrazyflieEnv(num_envs=1, use_native_step=True)
    native_env.room = room
    with pytest.raises(ValueError, match="does not support BoxRoom interior obstacles"):
        native_env.step(np.zeros((1, 4), dtype=np.float32))
