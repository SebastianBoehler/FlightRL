from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from flightrl.puffer4_config import Puffer4ExportSettings, render_puffer4_ini
from flightrl.puffer4_door_sections import build_fixed_door_teacher_sections


DOOR_NATIVE_FILES = (
    "native_sixdof.c",
    "native_sixdof.h",
    "native_sixdof_context.inc",
    "native_sixdof_observation.inc",
    "native_sixdof_step.inc",
    "native_sixdof_setpoint.c",
    "native_sixdof_setpoint.h",
    "native_door_action.c",
    "native_door_action.h",
    "native_door_mission.c",
    "native_door_mission.h",
    "native_door_episode_rng.c",
    "native_door_episode_rng.h",
    "native_door_episode_groups.inc",
    "native_door_domain.inc",
    "native_door_proprio.c",
    "native_door_proprio.h",
    "native_door_detector.c",
    "native_door_detector.h",
    "native_door_self_mask.c",
    "native_door_self_mask.h",
    "native_sixdof_vision.c",
    "native_sixdof_vision.h",
    "native_sixdof_vision_surfaces.inc",
    "native_door_scene.c",
    "native_door_scene.h",
    "native_door_scene_coverage.inc",
    "native_door_teacher.c",
    "native_door_teacher.h",
    "native_door_env_config.inc",
    "native_door_env_types.inc",
)


@dataclass(frozen=True, slots=True)
class DoorTeacherExportResult:
    env_name: str
    env_dir: Path
    config_path: Path


def export_fixed_door_assets(
    pufferlib_root: str | Path,
    settings: Puffer4ExportSettings,
) -> DoorTeacherExportResult:
    root = Path(pufferlib_root).expanduser().resolve()
    env_dir = root / "ocean" / settings.env_name
    config_path = root / "config" / f"{settings.env_name}.ini"
    env_dir.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    native_dir = Path(__file__).resolve().parent / "native"
    for filename in DOOR_NATIVE_FILES:
        shutil.copy2(native_dir / filename, env_dir / filename)
    dimensions = (
        "#define SIXDOF_VISION_WIDTH 64\n"
        "#define SIXDOF_VISION_HEIGHT 48\n"
    )
    binding = dimensions + (native_dir / "native_door_env_binding.c").read_text()
    (env_dir / "binding.c").write_text(binding)
    config_path.write_text(
        render_puffer4_ini(build_fixed_door_teacher_sections(settings))
    )
    return DoorTeacherExportResult(settings.env_name, env_dir, config_path)
