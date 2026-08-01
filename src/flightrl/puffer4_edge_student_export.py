from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from flightrl.puffer4_config import Puffer4ExportSettings, render_puffer4_ini
from flightrl.puffer4_door_export import DOOR_NATIVE_FILES
from flightrl.puffer4_edge_student_sections import build_edge_student_sections


EDGE_STUDENT_NATIVE_FILES = (
    *DOOR_NATIVE_FILES,
    "native_edge_student_action.c",
    "native_edge_student_action.h",
    "native_edge_student_observation.c",
    "native_edge_student_observation.h",
    "native_door_env_binding.c",
)


@dataclass(frozen=True, slots=True)
class EdgeStudentExportResult:
    env_name: str
    env_dir: Path
    config_path: Path


def render_edge_student_binding() -> str:
    native_dir = Path(__file__).resolve().parent / "native"
    binding_source = (native_dir / "native_door_env_binding.c").read_text()
    binding_source = binding_source.replace(
        '#include "native_door_lane.inc"',
        (native_dir / "native_door_lane.inc").read_text(),
    )
    return (
        "#define SIXDOF_VISION_WIDTH 64\n"
        "#define SIXDOF_VISION_HEIGHT 48\n"
        "#define FLIGHTRL_EDGE_STUDENT_LANE 1\n"
        + binding_source
    )


def export_edge_student_assets(
    pufferlib_root: str | Path,
    settings: Puffer4ExportSettings,
) -> EdgeStudentExportResult:
    root = Path(pufferlib_root).expanduser().resolve()
    env_dir = root / "ocean" / settings.env_name
    config_path = write_edge_student_config(root, settings)
    env_dir.mkdir(parents=True, exist_ok=True)
    native_dir = Path(__file__).resolve().parent / "native"
    for filename in EDGE_STUDENT_NATIVE_FILES:
        shutil.copy2(native_dir / filename, env_dir / filename)
    (env_dir / "binding.c").write_text(render_edge_student_binding())
    return EdgeStudentExportResult(settings.env_name, env_dir, config_path)


def write_edge_student_config(
    pufferlib_root: str | Path,
    settings: Puffer4ExportSettings,
) -> Path:
    root = Path(pufferlib_root).expanduser().resolve()
    config_path = root / "config" / f"{settings.env_name}.ini"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(render_puffer4_ini(build_edge_student_sections(settings)))
    return config_path
