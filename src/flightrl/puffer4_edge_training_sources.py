from __future__ import annotations

from pathlib import Path

from flightrl.evidence_scope import file_identity


ROOT = Path(__file__).resolve().parents[2]
EDGE_TRAINING_SOURCE_PATHS = {
    "script": ROOT / "scripts/train_puffer_edge_student.py",
    "artifact_paths": ROOT / "src/flightrl/artifact_paths.py",
    "trainer": ROOT / "src/flightrl/puffer4_edge_training.py",
    "policy": ROOT / "src/flightrl/puffer4_edge_policy.py",
    "training_data": ROOT / "src/flightrl/puffer4_edge_training_data.py",
    "training_report": ROOT / "src/flightrl/puffer4_edge_training_report.py",
    "selection_gate": ROOT / "src/flightrl/puffer4_edge_training_selection.py",
    "state_digest": ROOT / "src/flightrl/puffer4_edge_training_state.py",
    "native_identity": ROOT / "src/flightrl/puffer4_edge_native_build.py",
    "source_identity": Path(__file__).resolve(),
}


def edge_training_source_identity() -> dict[str, dict[str, str]]:
    return {
        name: file_identity(path)
        for name, path in EDGE_TRAINING_SOURCE_PATHS.items()
    }
