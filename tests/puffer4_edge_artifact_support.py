from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from flightrl.evidence_scope import file_identity
from flightrl.puffer4_door_runner import (
    BUILD_FINGERPRINT_SCHEMA_VERSION,
    BUILD_MODE,
    current_python_abi,
)
from flightrl.puffer4_edge_collection_evidence import (
    canonical_edge_environment_config,
)
from flightrl.puffer4_edge_checkpoint import (
    build_edge_checkpoint_payload,
    save_edge_checkpoint,
)
from flightrl.puffer4_edge_policy import EdgeNavigationActor
from flightrl.puffer4_edge_sequence import (
    EdgeSequenceDataset,
    edge_dataset_metadata,
    write_edge_sequence_dataset,
)
from flightrl.puffer4_edge_training import (
    EdgeTrainConfig,
    train_edge_student,
)
from flightrl.puffer4_edge_training_sources import edge_training_source_identity


ROOT = Path(__file__).resolve().parents[1]
ENV_NAME = "flightrl_edge_v3_door_student"
AUTHORITY = {
    "authority": "none",
    "deployment_authority": False,
    "hardware_approved": False,
    "controls_drone": False,
}


@dataclass(frozen=True)
class EdgeArtifacts:
    train: Path
    selection: Path
    final: Path
    training: Path
    checkpoint: Path
    evaluation: Path | None = None
    replay: Path | None = None
    native_root: Path | None = None


def native_build_fingerprint(
    root: Path,
    environment: str = ENV_NAME,
) -> dict:
    source = str((root / "ocean" / environment / "binding.c").resolve())
    sources = {source: "b" * 64}
    manifest = hashlib.sha256(
        json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": BUILD_FINGERPRINT_SCHEMA_VERSION,
        "env_name": environment,
        "build_mode": BUILD_MODE,
        "python_abi": current_python_abi(),
        "dependency_revision": {"git_commit": "a" * 40},
        "source_files_sha256": sources,
        "source_manifest_sha256": manifest,
        "source_manifest_sha256_before": manifest,
        "source_manifest_sha256_after": manifest,
        "extension": {
            "path": str((root / "pufferlib" / "_C.fixture.so").resolve()),
            "sha256": "c" * 64,
        },
    }


def write_sequence(path: Path, split: str, seed: int, fingerprint: dict) -> Path:
    steps, agents = 4, 2
    telemetry = np.zeros((steps, agents, 19), dtype=np.float32)
    telemetry[..., 8] = 1.0
    telemetry[..., 14] = 1.0
    bright = np.indices((steps, agents)).sum(axis=0) % 2 == 1
    actions = np.zeros((steps, agents, 4), dtype=np.float32)
    actions[..., 0][bright] = 0.8
    actions[..., 3][~bright] = 1.0
    grounding = np.zeros((steps, agents, 4), dtype=np.float32)
    grounding[bright] = (1.0, 0.25, -0.25, 0.4)
    resets = np.zeros((steps, agents), dtype=np.uint8)
    resets[0] = 1
    profile = {
        "obstacle_probability": 0.5,
        "camera_randomization": 1.0,
        "layout_diversity": 1.0,
    }
    dataset = EdgeSequenceDataset(
        packed_frames=np.where(bright[..., None], 255, 0)
        .astype(np.uint8)
        .repeat(1536, axis=2),
        telemetry=telemetry,
        target_ids=np.zeros((steps, agents), dtype=np.uint8),
        teacher_actions=actions,
        behavior_actions=actions.copy(),
        execution_student_mask=np.zeros(agents, dtype=np.uint8),
        grounding=grounding,
        resets=resets,
        dones=np.zeros((steps, agents), dtype=np.uint8),
        metadata=edge_dataset_metadata(
            split=split,
            base_seed=seed,
            appearance_seed=seed + 10_000,
            steps=steps,
            agents=agents,
            target_ids=(0,),
            environment=ENV_NAME,
            native_build_fingerprint=fingerprint,
            collection_profile=profile,
            environment_config=canonical_edge_environment_config(
                environment=ENV_NAME,
                agents=agents,
                base_seed=seed,
                appearance_seed=seed + 10_000,
                collection_profile=profile,
            ),
        ),
    )
    write_edge_sequence_dataset(path, dataset)
    return path


def training_report(
    actor: EdgeNavigationActor,
    train: Path,
    selection: Path,
) -> dict:
    from flightrl.puffer4_edge_sequence import load_edge_sequence_dataset

    train_dataset = load_edge_sequence_dataset(train)
    selection_dataset = load_edge_sequence_dataset(selection)
    trained, report = train_edge_student(
        train_dataset,
        selection_dataset,
        EdgeTrainConfig(epochs=8, tbptt_steps=2, learning_rate=5.0e-3),
    )
    actor.load_state_dict(trained.state_dict(), strict=True)
    report["datasets"] = {
        "train": file_identity(train),
        "selection": file_identity(selection),
    }
    report["native_build_fingerprint"] = train_dataset.metadata[
        "native_build_fingerprint"
    ]
    report["source_identity"] = edge_training_source_identity()
    return report


def checkpoint_artifacts(
    tmp_path: Path,
    fingerprint: dict | None = None,
) -> EdgeArtifacts:
    fingerprint = fingerprint or native_build_fingerprint(tmp_path / "native-build")
    train = write_sequence(tmp_path / "train.npz", "train", 11, fingerprint)
    selection = write_sequence(tmp_path / "selection.npz", "selection", 21, fingerprint)
    final = write_sequence(tmp_path / "final.npz", "final", 31, fingerprint)
    actor = EdgeNavigationActor(hidden_size=48)
    training = tmp_path / "training.json"
    training.write_text(json.dumps(training_report(actor, train, selection)) + "\n")
    checkpoint = tmp_path / "student.pt"
    payload = build_edge_checkpoint_payload(
        actor,
        trained_target_ids=[0],
        dataset=selection,
        training_report=training,
    )
    save_edge_checkpoint(payload, checkpoint)
    return EdgeArtifacts(train, selection, final, training, checkpoint)
