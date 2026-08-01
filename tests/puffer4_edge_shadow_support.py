from __future__ import annotations

import hashlib
import json
from pathlib import Path

from flightrl.evidence_scope import file_identity
from flightrl.puffer4_door_runner import (
    BUILD_FINGERPRINT_SCHEMA_VERSION,
    BUILD_MODE,
    current_python_abi,
    native_build_marker_path,
    native_extension_path,
    native_source_paths,
)
from flightrl.puffer4_config import Puffer4ExportSettings, render_puffer4_ini
from flightrl.puffer4_edge_contract import edge_policy_contract_report
from flightrl.puffer4_edge_evaluation_gate import (
    EDGE_EVALUATION_PROFILES,
    EDGE_EVALUATION_SCHEMA,
    collision_rate_upper_95,
    edge_student_gate,
)
from flightrl.puffer4_edge_replay import write_edge_passive_replay
from flightrl.puffer4_edge_student_export import EDGE_STUDENT_NATIVE_FILES
from flightrl.puffer4_edge_student_sections import build_edge_student_sections
from puffer4_edge_artifact_support import (
    AUTHORITY,
    ENV_NAME,
    ROOT,
    EdgeArtifacts,
    checkpoint_artifacts,
)


def shadow_artifacts(tmp_path: Path) -> EdgeArtifacts:
    native_root = tmp_path / "pufferlib"
    fingerprint = write_native_build(native_root)
    artifacts = checkpoint_artifacts(
        tmp_path,
        fingerprint=fingerprint,
    )
    evaluation = tmp_path / "held-out-evaluation.json"
    evaluation.write_text(
        json.dumps(evaluation_report(artifacts, fingerprint)) + "\n"
    )
    replay = tmp_path / "offline-replay.jsonl"
    write_edge_passive_replay(
        checkpoint=artifacts.checkpoint,
        dataset=artifacts.final,
        output=replay,
    )
    return EdgeArtifacts(
        artifacts.train,
        artifacts.selection,
        artifacts.final,
        artifacts.training,
        artifacts.checkpoint,
        evaluation,
        replay,
        native_root,
    )


def write_native_build(root: Path) -> dict:
    for path in native_source_paths(root, ENV_NAME, EDGE_STUDENT_NATIVE_FILES):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture source: {path.relative_to(root)}\n")
    extension = native_extension_path(root)
    extension.parent.mkdir(parents=True, exist_ok=True)
    extension.write_bytes(b"fixture native extension")
    manifest = {
        str(path.resolve()): _sha256(path)
        for path in native_source_paths(root, ENV_NAME, EDGE_STUDENT_NATIVE_FILES)
    }
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    fingerprint = {
        "schema_version": BUILD_FINGERPRINT_SCHEMA_VERSION,
        "env_name": ENV_NAME,
        "build_mode": BUILD_MODE,
        "python_abi": current_python_abi(),
        "dependency_revision": {"git_commit": "a" * 40},
        "source_files_sha256": manifest,
        "source_manifest_sha256": digest,
        "source_manifest_sha256_before": digest,
        "source_manifest_sha256_after": digest,
        "extension": {
            "path": str(extension.resolve()),
            "sha256": _sha256(extension),
        },
    }
    marker = native_build_marker_path(root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(fingerprint, sort_keys=True) + "\n")
    config = root / "config" / f"{ENV_NAME}.ini"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        render_puffer4_ini(
            build_edge_student_sections(
                Puffer4ExportSettings(
                    env_name=ENV_NAME,
                    total_agents=128,
                    num_buffers=1,
                    num_threads=8,
                    policy_hidden_size=48,
                    train_seed=17,
                )
            )
        )
    )
    return fingerprint


def evaluation_report(artifacts: EdgeArtifacts, fingerprint: dict) -> dict:
    profiles = {}
    for name, seed, appearance_seed, profile in EDGE_EVALUATION_PROFILES:
        metrics = passing_metrics(profile)
        profiles[name] = {
            "metrics": metrics,
            "gate": edge_student_gate(metrics, profile=profile),
            "seed": seed,
            "appearance_seed": appearance_seed,
            "profile": dict(profile),
            "steps": 6000,
            "agents": 128,
        }
    return {
        "schema": EDGE_EVALUATION_SCHEMA,
        "status": "complete",
        "scope": "desktop_simulation_held_out",
        "checkpoint_identity": file_identity(artifacts.checkpoint),
        "policy_contract_sha256": edge_policy_contract_report(hidden_size=48)[
            "sha256"
        ],
        "evaluated_target_ids": [0],
        "profiles": profiles,
        "gate": {"passed": True, "failures": []},
        "native_build_fingerprint": fingerprint,
        "environment_config_identity": file_identity(
            Path(fingerprint["extension"]["path"]).parent.parent
            / "config"
            / f"{ENV_NAME}.ini"
        ),
        "source_identity": {
            "script": file_identity(ROOT / "scripts/evaluate_puffer_edge_student.py"),
            "artifact_paths": file_identity(
                ROOT / "src/flightrl/artifact_paths.py"
            ),
            "evaluator": file_identity(
                ROOT / "src/flightrl/puffer4_edge_evaluation.py"
            ),
            "gate": file_identity(
                ROOT / "src/flightrl/puffer4_edge_evaluation_gate.py"
            ),
            "counts": file_identity(
                ROOT / "src/flightrl/puffer4_edge_evaluation_counts.py"
            ),
            "metrics": file_identity(
                ROOT / "src/flightrl/puffer4_edge_evaluation_metrics.py"
            ),
            "exporter": file_identity(
                ROOT / "src/flightrl/puffer4_edge_student_export.py"
            ),
            "sections": file_identity(
                ROOT / "src/flightrl/puffer4_edge_student_sections.py"
            ),
            "door_sections": file_identity(
                ROOT / "src/flightrl/puffer4_door_sections.py"
            ),
            "config": file_identity(ROOT / "src/flightrl/puffer4_config.py"),
            "native_identity": file_identity(
                ROOT / "src/flightrl/puffer4_edge_native_build.py"
            ),
        },
        **AUTHORITY,
    }


def passing_metrics(profile: dict[str, float]) -> dict[str, float]:
    randomized_camera = profile["camera_randomization"] > 0.0
    obstacle_probability = profile["obstacle_probability"]
    episodes = 256.0
    collision_rate = 2.0 / episodes
    success_rate = 244.0 / episodes
    metrics = {
        "score": success_rate,
        "episode_return": 5.0,
        "episode_length": 50.0,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "collision_rate_upper_95": collision_rate_upper_95(
            collision_rate,
            episodes,
        ),
        "outside_fov_success_fraction": 116.0 / episodes,
        "outside_fov_episode_fraction": 0.50,
        "outside_fov_observed_fraction": 125.0 / episodes,
        "observed_episode_fraction": 253.0 / episodes,
        "door_visible_fraction": 0.75,
        "low_light_episode_fraction": 0.25 if randomized_camera else 0.0,
        "low_light_success_fraction": 60.0 / episodes if randomized_camera else 0.0,
        "obstacle_episode_fraction": obstacle_probability,
        "obstacle_success_fraction": (
            success_rate if obstacle_probability == 1.0
            else 120.0 / episodes if obstacle_probability > 0.0
            else 0.0
        ),
        "scene_group_schema_version": 1.0,
        "n": episodes,
        "outside_fov_success_rate": 116.0 / 128.0,
        "outside_fov_episodes": 128.0,
        "episodes": episodes,
        "action_rmse": 0.10,
        "door_action_rmse": 0.10,
        "reset_action_rmse": 0.15,
        "reset_door_action_rmse": 0.20,
        "reset_samples": 256.0,
        "lateral_action_abs_mean": 0.02,
        "vertical_action_abs_mean": 0.02,
        "lateral_action_abs_max": 0.10,
        "vertical_action_abs_max": 0.10,
        "action_saturation_fraction": 0.01,
        "grounding_visibility_precision": 0.95,
        "grounding_visibility_recall": 0.95,
        "grounding_visible_box_mae": 0.05,
        "grounding_visible_samples": 384_000.0,
        "grounding_absent_samples": 384_000.0,
        "hidden_min": 0.0,
        "hidden_max": 2.0,
    }
    for group in ("layout_family", "door_face"):
        for index in range(1, 4):
            metrics[f"{group}_{index}_episode_fraction"] = 0.25
            metrics[f"{group}_{index}_success_fraction"] = 61.0 / episodes
    return metrics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
