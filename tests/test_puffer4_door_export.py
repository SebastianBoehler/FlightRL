from __future__ import annotations

from pathlib import Path

from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_door_export import (
    DOOR_NATIVE_FILES,
    export_fixed_door_assets,
)


def test_privileged_teacher_export_has_no_goal_or_live_authority(
    tmp_path: Path,
) -> None:
    result = export_fixed_door_assets(
        tmp_path / "PufferLib-4",
        Puffer4ExportSettings(
            env_name="flightrl_fixed_door_test",
            total_agents=64,
            num_buffers=1,
            num_threads=4,
            policy_hidden_size=48,
            train_seed=11,
        ),
    )
    binding = (result.env_dir / "binding.c").read_text()
    config = result.config_path.read_text()

    assert "#define OBS_SIZE SIXDOF_DOOR_OBS_DIM" in binding
    assert "#define NUM_ATNS 2" in binding
    assert "flightrl_sixdof_door_observation_scene" in binding
    assert "flightrl_door_control_action(" in binding
    assert "env->physics[SIXDOF_PHYS_MAX_RATE_YAW]" in binding
    assert "flightrl_sixdof_setpoint_actions_batch" in binding
    assert "waypoint_residual" not in binding
    assert "SIXDOF_DOOR_POLICY_OBS_DIM" in binding
    assert "native_door_proprio.c" in DOOR_NATIVE_FILES
    assert "native_door_detector.c" in DOOR_NATIVE_FILES
    assert "native_door_episode_rng.c" in DOOR_NATIVE_FILES
    assert "native_door_episode_rng.h" in DOOR_NATIVE_FILES
    assert "native_door_episode_groups.inc" in DOOR_NATIVE_FILES
    assert "native_door_mission.c" in DOOR_NATIVE_FILES
    assert "native_door_mission.h" in DOOR_NATIVE_FILES
    assert "native_door_self_mask.c" in DOOR_NATIVE_FILES
    assert "native_sixdof_vision_surfaces.inc" in DOOR_NATIVE_FILES
    assert "native_door_scene_coverage.inc" in DOOR_NATIVE_FILES
    assert "flightrl_door_detector_update" in binding
    assert "flightrl_door_teacher_action(\n        env->position" in binding
    assert "flightrl_door_detector_teacher_action(&env->detector" not in binding
    assert "flightrl_door_mission_step(" in binding
    assert "env->mission.target_standoff_m" in binding
    assert (
        "fminf(env->mission.planar_position_tolerance_m, "
        "env->mission.standoff_tolerance_m)" in binding
    )
    assert "encoder = DefaultEncoder" in config
    assert "network = MLP" in config
    assert "total_timesteps = 0" in config
    assert "total_agents = 64" in config
    assert "camera_mask = 0" in config
    assert "domain_randomization = 1" in config
    assert "layout_diversity = 0" in config
    assert "camera_randomization = 0" in config
    assert "obstacle_probability = 0.5" in config
    assert "max_episode_steps = 2600" in config
    assert "success_radius_m" not in config
    assert "mission_target_standoff_m = 0.8" in config
    assert "mission_planar_position_tolerance_m = 0.1" in config
    assert "mission_vertical_position_tolerance_m = 0.1" in config
    assert "mission_standoff_tolerance_m = 0.08" in config
    assert "mission_yaw_alignment_tolerance_rad = 0.174532925199" in config
    assert "mission_max_horizontal_speed_m_s = 0.08" in config
    assert "mission_max_vertical_speed_m_s = 0.05" in config
    assert "mission_max_yaw_rate_rad_s = 0.0872664625997" in config
    assert "mission_dwell_steps = 33" in config
    assert "max_horizontal_speed_m_s = 0.25" in config
    assert "max_vertical_speed_m_s = 0.1" in config
    assert "max_yawrate_deg_s = 45" in config
    for filename in DOOR_NATIVE_FILES:
        assert (result.env_dir / filename).exists()
