from __future__ import annotations

from collections.abc import Mapping
import configparser
import json
from math import isfinite
from pathlib import Path

from flightrl.artifact_identity import canonical_json_bytes, file_identity, sha256_payload
from flightrl.puffer4_config import Puffer4ExportSettings, render_puffer4_ini
from flightrl.puffer4_edge_contract import edge_policy_contract_report
from flightrl.puffer4_edge_execution import edge_execution_provenance
from flightrl.puffer4_edge_native_build import (
    canonical_edge_native_build_fingerprint,
)
from flightrl.puffer4_edge_student_sections import build_edge_student_sections


ROOT = Path(__file__).resolve().parents[2]
EDGE_DATASET_SCHEMA = "flightrl.edge_v3.sequence_dataset.v5"
_SPLITS = ("train", "selection", "final")
_PROFILE_FIELDS = {
    "obstacle_probability",
    "camera_randomization",
    "layout_diversity",
}
_FIELDS = {
    "schema", "split", "base_seed", "appearance_seed", "steps", "agents",
    "target_ids", "environment", "execution_policy",
    "execution_checkpoint_identity", "execution_mix",
    "policy_contract_sha256", "native_build_fingerprint",
    "collection_profile", "environment_config",
    "environment_config_sha256", "collection_source_identity",
}
_SOURCES = {
    "collector": ROOT / "scripts/build_puffer_edge_dataset.py",
    "artifact_paths": ROOT / "src/flightrl/artifact_paths.py",
    "adapter": ROOT / "src/flightrl/puffer4_edge_dataset.py",
    "collection_arrays": ROOT
    / "src/flightrl/puffer4_edge_collection_arrays.py",
    "execution": ROOT / "src/flightrl/puffer4_edge_execution.py",
    "episode_provenance": ROOT
    / "src/flightrl/puffer4_edge_episode_provenance.py",
    "dagger": ROOT / "src/flightrl/puffer4_edge_dagger.py",
    "sequence": ROOT / "src/flightrl/puffer4_edge_sequence.py",
    "collection_evidence": ROOT
    / "src/flightrl/puffer4_edge_collection_evidence.py",
    "exporter": ROOT / "src/flightrl/puffer4_edge_student_export.py",
    "sections": ROOT / "src/flightrl/puffer4_edge_student_sections.py",
    "door_sections": ROOT / "src/flightrl/puffer4_door_sections.py",
    "config": ROOT / "src/flightrl/puffer4_config.py",
    "runner": ROOT / "src/flightrl/puffer4_door_runner.py",
    "mission": ROOT / "src/flightrl/puffer4_door_mission.py",
    "native_identity": ROOT / "src/flightrl/puffer4_edge_native_build.py",
}


def build_edge_dataset_metadata(
    *,
    split: str,
    base_seed: int,
    appearance_seed: int,
    steps: int,
    agents: int,
    target_ids: tuple[int, ...],
    environment: str,
    native_build_fingerprint: Mapping,
    collection_profile: Mapping[str, float],
    environment_config: Mapping[str, int | float],
    execution_policy: str = "privileged_teacher",
    execution_checkpoint_identity: dict[str, str] | None = None,
    execution_student_fraction: float | None = None,
    execution_mix_seed: int | None = None,
) -> dict:
    config = dict(environment_config)
    metadata = {
        "schema": EDGE_DATASET_SCHEMA,
        "split": split,
        "base_seed": base_seed,
        "appearance_seed": appearance_seed,
        "steps": steps,
        "agents": agents,
        "target_ids": list(target_ids),
        "environment": environment,
        **edge_execution_provenance(
            execution_policy,
            execution_checkpoint_identity,
            split=split,
            agents=agents,
            student_fraction=execution_student_fraction,
            mix_seed=execution_mix_seed,
        ),
        "policy_contract_sha256": edge_policy_contract_report(
            hidden_size=48
        )["sha256"],
        "native_build_fingerprint": canonical_edge_native_build_fingerprint(
            native_build_fingerprint
        ),
        "collection_profile": dict(collection_profile),
        "environment_config": config,
        "environment_config_sha256": edge_environment_config_sha256(config),
        "collection_source_identity": edge_collection_source_identity(),
    }
    require_edge_collection_metadata(metadata)
    return metadata


def canonical_edge_environment_config(
    *,
    environment: str,
    agents: int,
    base_seed: int,
    appearance_seed: int,
    collection_profile: Mapping[str, float],
) -> dict[str, int | float]:
    profile = _require_profile(collection_profile)
    settings = Puffer4ExportSettings(
        env_name=environment,
        total_agents=agents,
        num_buffers=1,
        num_threads=min(agents, 8),
        policy_hidden_size=48,
        train_seed=17,
    )
    parser = configparser.ConfigParser()
    parser.read_string(render_puffer4_ini(build_edge_student_sections(settings)))
    config = {
        name: _parse_ini_value(value)
        for name, value in parser["env"].items()
    }
    config.update(
        {
            "seed": base_seed,
            "appearance_seed": appearance_seed,
            **profile,
            "camera_mask": 0.0,
        }
    )
    return config


def edge_environment_config_sha256(value: object) -> str:
    return sha256_payload(value)


def edge_collection_source_identity() -> dict[str, dict[str, str]]:
    return {name: file_identity(path) for name, path in _SOURCES.items()}


def require_edge_collection_metadata(metadata: object) -> None:
    if not isinstance(metadata, dict) or set(metadata) != _FIELDS:
        raise ValueError("edge dataset metadata fields are incompatible")
    if metadata["schema"] != EDGE_DATASET_SCHEMA or metadata["split"] not in _SPLITS:
        raise ValueError("edge dataset schema or split is incompatible")
    for name in ("base_seed", "appearance_seed"):
        if type(metadata[name]) is not int or not 0 <= metadata[name] < 2**32:
            raise ValueError(f"edge dataset {name} must be uint32")
    for name in ("steps", "agents"):
        if type(metadata[name]) is not int or metadata[name] <= 0:
            raise ValueError(f"edge dataset {name} must be positive")
    if metadata["target_ids"] != [0]:
        raise ValueError("first edge dataset must be explicitly door-only")
    _require_execution(metadata)
    environment = metadata["environment"]
    if not isinstance(environment, str) or not environment:
        raise ValueError("edge dataset environment is invalid")
    expected_contract = edge_policy_contract_report(hidden_size=48)["sha256"]
    if metadata["policy_contract_sha256"] != expected_contract:
        raise ValueError("edge dataset policy contract SHA-256 is incompatible")
    fingerprint = canonical_edge_native_build_fingerprint(
        metadata["native_build_fingerprint"]
    )
    if fingerprint["env_name"] != environment:
        raise ValueError("edge dataset native build environment does not match")
    profile = _require_profile(metadata["collection_profile"])
    config = metadata["environment_config"]
    expected = canonical_edge_environment_config(
        environment=environment,
        agents=metadata["agents"],
        base_seed=metadata["base_seed"],
        appearance_seed=metadata["appearance_seed"],
        collection_profile=profile,
    )
    if canonical_json_bytes(config) != canonical_json_bytes(expected):
        raise ValueError("edge dataset environment config is not canonical")
    digest = edge_environment_config_sha256(config)
    if metadata["environment_config_sha256"] != digest:
        raise ValueError("edge dataset environment config SHA-256 does not reproduce")
    if metadata["collection_source_identity"] != edge_collection_source_identity():
        raise ValueError("edge dataset collection source identity does not match")


def _require_execution(metadata: Mapping) -> None:
    mix = metadata["execution_mix"]
    dagger = metadata["execution_policy"] == "dagger_student"
    student = mix.get("student") if isinstance(mix, dict) and dagger else None
    seed = mix.get("seed") if isinstance(mix, dict) and dagger else None
    expected = edge_execution_provenance(
        metadata["execution_policy"],
        metadata["execution_checkpoint_identity"],
        split=metadata["split"],
        agents=metadata["agents"],
        student_fraction=student,
        mix_seed=seed,
    )
    if any(metadata[name] != value for name, value in expected.items()):
        raise ValueError("edge dataset execution mix is not canonical")


def _require_profile(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != _PROFILE_FIELDS:
        raise ValueError("edge dataset collection profile is invalid")
    profile = {}
    for name in _PROFILE_FIELDS:
        item = value[name]
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not isfinite(float(item))
            or not 0.0 <= float(item) <= 1.0
        ):
            raise ValueError("edge dataset collection profile is invalid")
        profile[name] = float(item)
    return profile


def _parse_ini_value(value: str) -> int | float:
    parsed = json.loads(value)
    if isinstance(parsed, bool) or not isinstance(parsed, (int, float)):
        raise ValueError("edge environment config contains a nonnumeric value")
    return parsed
