from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Iterable, Sequence

import torch

from flightrl.puffer4_door_export import DOOR_NATIVE_FILES
from flightrl.puffer4_door_provenance import (
    build_door_run_provenance,
    build_file_manifest,
)
from flightrl.puffer4_door_runner import (
    native_build_marker_path,
    native_extension_path,
)


@dataclass(frozen=True, slots=True)
class FixedDoorEvaluationProvenanceCapture:
    command: tuple[str, ...]
    started_at_utc: str
    started_perf_ns: int
    flightrl_root: Path
    entrypoint: Path
    source_manifest: dict[str, str]


def begin_fixed_door_evaluation_provenance(
    *,
    command: Sequence[str],
    flightrl_root: Path,
    entrypoint: Path,
) -> FixedDoorEvaluationProvenanceCapture:
    root = flightrl_root.resolve()
    resolved_entrypoint = entrypoint.resolve()
    paths = fixed_door_evaluation_source_paths(root, resolved_entrypoint)
    return FixedDoorEvaluationProvenanceCapture(
        command=tuple(command),
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        started_perf_ns=perf_counter_ns(),
        flightrl_root=root,
        entrypoint=resolved_entrypoint,
        source_manifest=build_file_manifest(root, paths),
    )


def fixed_door_evaluation_source_paths(
    root: Path,
    entrypoint: Path,
) -> tuple[Path, ...]:
    root = root.resolve()
    native_root = root / "src" / "flightrl" / "native"
    paths = {
        entrypoint.resolve(),
        *root.glob("src/flightrl/puffer4_door*.py"),
        *(
            path
            for name in (*DOOR_NATIVE_FILES, "native_door_env_binding.c")
            if (path := native_root / name).is_file()
        ),
    }
    return tuple(sorted(paths, key=lambda path: str(path)))


def fixed_door_generated_paths(
    puffer_root: Path,
    env_name: str,
) -> tuple[Path, ...]:
    root = puffer_root.resolve()
    env_dir = root / "ocean" / env_name
    return (
        root / "config" / f"{env_name}.ini",
        env_dir / "binding.c",
        *(env_dir / name for name in DOOR_NATIVE_FILES),
        native_extension_path(root),
        root / "pufferlib" / "torch_pufferl.py",
        native_build_marker_path(root),
    )


def build_fixed_door_evaluation_provenance(
    *,
    command: Sequence[str],
    started_at_utc: str,
    elapsed_wall_s: float,
    lineage_report: Path,
    flightrl_root: Path,
    flightrl_source_sha256: dict[str, str],
    puffer_root: Path,
    generated_files: Iterable[Path],
    native_build_fingerprint: dict,
) -> dict:
    provenance = build_door_run_provenance(
        command=command,
        started_at_utc=started_at_utc,
        elapsed_wall_s=elapsed_wall_s,
        source_report=lineage_report,
        flightrl_root=flightrl_root,
        flightrl_source_sha256=flightrl_source_sha256,
        puffer_root=puffer_root,
        generated_files=generated_files,
        native_build_fingerprint=native_build_fingerprint,
    )
    provenance["torch_version"] = str(torch.__version__)
    return provenance


def write_fixed_door_evaluation_report(
    *,
    report: dict,
    output: Path,
    capture: FixedDoorEvaluationProvenanceCapture,
    lineage_report: Path,
    puffer_root: Path,
    env_name: str,
    native_build_fingerprint: dict,
) -> dict:
    if "evaluation_provenance" in report:
        raise ValueError("evaluation report already contains provenance")
    provenance = build_fixed_door_evaluation_provenance(
        command=capture.command,
        started_at_utc=capture.started_at_utc,
        elapsed_wall_s=(perf_counter_ns() - capture.started_perf_ns) / 1.0e9,
        lineage_report=lineage_report.resolve(),
        flightrl_root=capture.flightrl_root,
        flightrl_source_sha256=capture.source_manifest,
        puffer_root=puffer_root.resolve(),
        generated_files=fixed_door_generated_paths(puffer_root, env_name),
        native_build_fingerprint=native_build_fingerprint,
    )
    current = build_file_manifest(
        capture.flightrl_root,
        fixed_door_evaluation_source_paths(
            capture.flightrl_root,
            capture.entrypoint,
        ),
    )
    if current != capture.source_manifest:
        raise RuntimeError("FlightRL evaluation source manifest changed during run")
    resolved_report = dict(report)
    resolved_report["evaluation_provenance"] = provenance
    payload = json.dumps(resolved_report, indent=2, sort_keys=True) + "\n"
    with output.resolve().open("x") as handle:
        handle.write(payload)
    return resolved_report
