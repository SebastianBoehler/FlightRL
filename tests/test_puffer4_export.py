from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys

import pytest

from flightrl import load_config
from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_export import PUFFER4_NATIVE_FILES, export_puffer4_assets
from flightrl.puffer4_runtime import ensure_puffer_build_matches, normalize_puffer_args, puffer_subprocess_env
from flightrl.puffer4_sixdof_export import SIXDOF_NATIVE_FILES, export_sixdof_puffer4_assets, render_sixdof_puffer4_binding


ROOT = Path(__file__).resolve().parents[1]


def test_export_puffer4_assets_writes_binding_and_ini(tmp_path: Path) -> None:
    pufferlib_root = tmp_path / "PufferLib-4.0"
    (pufferlib_root / "config").mkdir(parents=True)
    (pufferlib_root / "ocean").mkdir(parents=True)

    config = load_config(ROOT / "configs" / "tasks" / "hover.toml")
    result = export_puffer4_assets(config, pufferlib_root)

    binding_path = result.env_dir / "binding.c"
    ini_text = result.config_path.read_text()
    binding_text = binding_path.read_text()

    assert result.env_name == "flightrl"
    assert binding_path.exists()
    assert "#define OBS_SIZE" in binding_text
    assert f"#define OBS_SIZE {config.observation_dim}" in binding_text
    assert f"#define NUM_ATNS {config.action_dim}" in binding_text
    assert 'dict_get(kwargs, "waypoint_7_z")' in binding_text
    assert "[base]" in ini_text
    assert "env_name = flightrl" in ini_text
    assert "total_agents = 256" in ini_text
    assert "replay_ratio = 2" in ini_text
    assert "horizon = 32" in ini_text
    assert "minibatch_size = 1024" in ini_text

    for filename in PUFFER4_NATIVE_FILES:
        assert (result.env_dir / filename).exists()
    assert (result.env_dir / "native_sixdof_step.inc").exists()


def test_export_puffer4_assets_supports_hover_command_shape(tmp_path: Path) -> None:
    pufferlib_root = tmp_path / "PufferLib-4.0"
    config = load_config(ROOT / "configs" / "tasks" / "crazyflie_hover_command.toml")
    result = export_puffer4_assets(config, pufferlib_root, settings=Puffer4ExportSettings(env_name="flightrl_hover_command"))
    binding_text = (result.env_dir / "binding.c").read_text()
    ini_text = result.config_path.read_text()

    assert config.action_dim == 4
    assert "#define NUM_ATNS 4" in binding_text
    assert "action_mode = 3" in ini_text
    assert "max_velocity = 0.45" in ini_text
    assert "env_name = flightrl_hover_command" in ini_text


def test_export_puffer4_assets_respects_overrides(tmp_path: Path) -> None:
    pufferlib_root = tmp_path / "PufferLib-4.0"
    (pufferlib_root / "config").mkdir(parents=True)
    (pufferlib_root / "ocean").mkdir(parents=True)

    config = load_config(ROOT / "configs" / "tasks" / "reach.toml")
    settings = Puffer4ExportSettings(
        env_name="flightrl_reach",
        total_agents=384,
        num_buffers=6,
        num_threads=3,
        policy_hidden_size=192,
        policy_num_layers=4,
        train_seed=99,
    )
    result = export_puffer4_assets(config, pufferlib_root, settings=settings)
    ini_text = result.config_path.read_text()

    assert result.env_name == "flightrl_reach"
    assert "env_name = flightrl_reach" in ini_text
    assert "total_agents = 384" in ini_text
    assert "num_buffers = 6" in ini_text
    assert "num_threads = 3" in ini_text
    assert "hidden_size = 192" in ini_text
    assert "num_layers = 4" in ini_text
    assert "seed = 99" in ini_text


def test_export_sixdof_puffer4_assets_writes_native_ocean_env(tmp_path: Path) -> None:
    pufferlib_root = tmp_path / "PufferLib-4.0"
    result = export_sixdof_puffer4_assets(
        pufferlib_root,
        settings=Puffer4ExportSettings(env_name="flightrl_sixdof_test", total_agents=2048, num_buffers=4, train_seed=17),
    )
    binding_text = (result.env_dir / "binding.c").read_text()
    ini_text = result.config_path.read_text()

    assert result.env_name == "flightrl_sixdof_test"
    assert "#define OBS_SIZE 28" in binding_text
    assert "#define NUM_ATNS 4" in binding_text
    assert "flightrl_sixdof_step_env_context_batch" in binding_text
    assert "native_sixdof.c" in binding_text
    assert "env_name = flightrl_sixdof_test" in ini_text
    assert "total_agents = 2048" in ini_text
    assert "num_buffers = 4" in ini_text
    assert "dt = 0.01" in ini_text
    assert "room_x_min = -2" in ini_text
    assert "range_noise_std_m = 0" in ini_text
    assert "action_lag_s = 0" in ini_text
    assert "task_id = 0" in ini_text
    assert "reward_mode = 0" in ini_text
    assert "near_wall_probability = 0" in ini_text
    assert "reset_z_min = 0.35" in ini_text
    assert "reset_z_max = 0.9" in ini_text
    assert 'dict_get(kwargs, "room_x_min")' in binding_text
    assert 'dict_get(kwargs, "range_noise_std_m")' in binding_text
    assert 'dict_get(kwargs, "task_id")' in binding_text
    assert 'dict_get(kwargs, "near_wall_probability")' in binding_text
    assert 'dict_get(kwargs, "reset_z_min")' in binding_text
    assert "apply_observation_profile" in binding_text
    assert "apply_reset_profile" in binding_text
    for filename in SIXDOF_NATIVE_FILES:
        assert (result.env_dir / filename).exists()


def test_export_sixdof_puffer4_assets_supports_task_reward_mode(tmp_path: Path) -> None:
    pufferlib_root = tmp_path / "PufferLib-4.0"
    result = export_sixdof_puffer4_assets(
        pufferlib_root,
        settings=Puffer4ExportSettings(env_name="flightrl_obstacle", task="obstacle_avoidance", reward_mode="live_clearance", reset_profile="obstacle_close_live"),
    )
    binding_text = (result.env_dir / "binding.c").read_text()
    ini_text = result.config_path.read_text()

    assert "task_id = 1" in ini_text
    assert "reward_mode = 4" in ini_text
    assert "near_wall_probability = 0.65" in ini_text
    assert "reset_z_min = 0.44" in ini_text
    assert "target_z_max = 0.6" in ini_text
    assert "target_xy_offset_abs = 0.45" in ini_text
    assert "task_position_error" in binding_text
    assert "flightrl_sixdof_step_env_context_batch" in binding_text


def test_render_sixdof_binding_is_ocean_shaped() -> None:
    binding = render_sixdof_puffer4_binding()
    assert "#define Env FlightRLSixDofEnv" in binding
    assert '#include "vecenv.h"' in binding
    assert "static void c_reset" in binding
    assert "static void c_step" in binding


def test_sixdof_puffer_export_report_cli_writes_evidence(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_sixdof_puffer_export_report.py",
            "--pufferlib-root",
            str(tmp_path / "PufferLib-4.0"),
            "--env-name",
            "flightrl_sixdof_evidence",
            "--output",
            str(output),
            "--total-agents",
            "128",
            "--num-buffers",
            "4",
            "--num-threads",
            "2",
            "--hidden-size",
            "64",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text())
    assert "puffer_export=" in result.stdout
    assert report["passed"] is True
    assert report["config"]["vec"]["total_agents"] == "128"
    assert report["config"]["vec"]["num_threads"] == "2"
    assert report["config"]["env"]["task_id"] == "0"
    assert report["config"]["env"]["reward_mode"] == "0"
    assert report["config"]["env"]["near_wall_probability"] == "0"
    assert report["config"]["env"]["reset_z_min"] == "0.35"
    assert report["files"]["binding.c"]["required_tokens"]["#define OBS_SIZE 28"]
    assert output.with_suffix(".md").exists()


def test_cpu_puffer_runtime_sets_openmp_guard() -> None:
    assert normalize_puffer_args((), "cpu") == ["--slowly"]
    env = puffer_subprocess_env("cpu", ())
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["KMP_DUPLICATE_LIB_OK"] == "TRUE"


def test_puffer_runtime_loads_project_wandb_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".secrets").mkdir()
    (tmp_path / ".secrets" / "wandb.env").write_text("WANDB_API_KEY=fresh\nWANDB_PROJECT=FlightRL\n")
    monkeypatch.setenv("WANDB_API_KEY", "stale")

    env = puffer_subprocess_env("cpu", ())

    assert env["WANDB_API_KEY"] == "fresh"
    assert env["WANDB_PROJECT"] == "FlightRL"


def test_no_build_puffer_runtime_rejects_mismatched_compiled_env(monkeypatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="flightrl_old\n")

    monkeypatch.setattr("flightrl.puffer4_runtime.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="flightrl_old"):
        ensure_puffer_build_matches(tmp_path, "flightrl_new", no_build=True)
