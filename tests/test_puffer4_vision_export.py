from __future__ import annotations

from pathlib import Path

from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_vision_export import (
    VISUAL_NATIVE_FILES,
    export_visual_puffer4_assets,
    render_visual_puffer4_binding,
)


def test_visual_export_writes_native_camera_navigation_env(tmp_path: Path) -> None:
    result = export_visual_puffer4_assets(
        tmp_path / "PufferLib-4",
        settings=Puffer4ExportSettings(
            env_name="flightrl_visual_test",
            total_agents=96,
            num_buffers=1,
            num_threads=4,
            policy_hidden_size=64,
        ),
    )
    binding = (result.env_dir / "binding.c").read_text()
    config = result.config_path.read_text()

    assert "#define OBS_SIZE SIXDOF_VISION_OBS_DIM" in binding
    assert "#define NUM_ATNS 4" in binding
    assert "flightrl_sixdof_visual_observation_scene" in binding
    assert "flightrl_sixdof_waypoint_residual_actions_batch" in binding
    assert "obstacle_hit" in binding
    assert "PROGRESS_REWARD_SCALE" in binding
    assert "ACTION_COST_SCALE" in binding
    assert "AVOIDANCE_REWARD_SCALE" in binding
    assert "flightrl_sixdof_avoidance_alignment" in binding
    assert "env->obstacle_probability" in binding
    assert "0x9e3779b9u * (env_index + 1u)" in binding
    assert "env->terminal = collision || success" in binding
    assert "int success = distance <= env->success_radius;" in binding
    assert "env_name = flightrl_visual_test" in config
    assert "total_agents = 96" in config
    assert "control_dt = 0.0153846153846" in config
    assert "physics_substeps = 2" in config
    assert "max_horizontal_speed_m_s = 0.45" in config
    assert "obstacle_probability = 0.75" in config
    assert "navigation_residual_scale = 0.6" in config
    assert "waypoint_slowdown_distance_m = 0.55" in config
    assert "learning_rate = 0.003" in config
    assert "encoder = FlightRLVisionEncoder" in config
    assert "network = MinGRU" in config
    for filename in VISUAL_NATIVE_FILES:
        assert (result.env_dir / filename).exists()


def test_visual_export_defaults_are_memory_bounded() -> None:
    binding = render_visual_puffer4_binding()

    assert "uint8_t previous_frame[SIXDOF_VISION_PIXELS]" in binding
    assert "float state_observation[28]" in binding
    assert "float* observations" in binding
