from __future__ import annotations

from pathlib import Path

from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_door_export import export_fixed_door_assets
from flightrl.puffer4_edge_student_export import (
    EDGE_STUDENT_NATIVE_FILES,
    export_edge_student_assets,
)
from flightrl.puffer4_edge_student_sections import build_edge_student_sections


def _settings(name: str) -> Puffer4ExportSettings:
    return Puffer4ExportSettings(
        env_name=name,
        total_agents=32,
        num_buffers=1,
        num_threads=4,
        train_seed=17,
    )


def test_edge_student_export_is_a_separate_exact_four_action_lane(
    tmp_path: Path,
) -> None:
    result = export_edge_student_assets(
        tmp_path / "PufferLib-4",
        _settings("flightrl_edge_student_test"),
    )
    binding = (result.env_dir / "binding.c").read_text()
    shared_binding = (result.env_dir / "native_door_env_binding.c").read_text()
    observation_header = (
        result.env_dir / "native_edge_student_observation.h"
    ).read_text()
    config = result.config_path.read_text()

    assert "#define FLIGHTRL_EDGE_STUDENT_LANE 1" in binding
    assert "#define OBS_TENSOR_T FloatTensor" in binding
    assert "#define NUM_ATNS 4" in binding
    assert "#define ACT_SIZES {1, 1, 1, 1}" in binding
    assert "#define NUM_ATNS 4" in shared_binding
    assert "#define FLIGHTRL_EDGE_STUDENT_OBS_DIM" in observation_header
    assert "max_horizontal_speed_m_s = 0.25" in config
    assert "max_vertical_speed_m_s = 0.15" in config
    assert "max_yawrate_deg_s = 45" in config
    assert "hidden_size = 48" in config
    assert "total_timesteps = 0" in config
    for filename in EDGE_STUDENT_NATIVE_FILES:
        assert (result.env_dir / filename).is_file()


def test_edge_student_sections_keep_training_tail_out_of_puffer_training() -> None:
    sections = build_edge_student_sections(_settings("edge_student"))

    assert sections["base"]["env_name"] == "edge_student"
    assert sections["train"]["total_timesteps"] == 0
    assert sections["policy"]["hidden_size"] == 48
    assert sections["env"]["max_vertical_speed_m_s"] == 0.15


def test_existing_privileged_teacher_export_does_not_enable_student_lane(
    tmp_path: Path,
) -> None:
    result = export_fixed_door_assets(
        tmp_path / "PufferLib-4",
        _settings("flightrl_teacher_test"),
    )
    binding = (result.env_dir / "binding.c").read_text()

    assert "#define FLIGHTRL_EDGE_STUDENT_LANE 1" not in binding
    assert "#define NUM_ATNS 2" in binding
    assert "#define ACT_SIZES {1, 1}" in binding
