from __future__ import annotations

from pathlib import Path

from flightrl.artifact_paths import require_distinct_artifact_paths
from flightrl.evidence_scope import file_identity


def capture_inputs(
    checkpoint: str | Path,
    dataset: str | Path,
    output: str | Path,
) -> tuple[dict[str, dict[str, str]], Path]:
    paths = require_distinct_artifact_paths(
        checkpoint=checkpoint,
        dataset=dataset,
        output=output,
    )
    identities = {
        "checkpoint_identity": file_identity(paths["checkpoint"]),
        "dataset_identity": file_identity(paths["dataset"]),
    }
    return identities, paths["output"]


def require_unchanged(
    identities: dict[str, dict[str, str]],
    stage: str,
) -> None:
    for field in ("checkpoint_identity", "dataset_identity"):
        label = field.removesuffix("_identity")
        identity = identities[field]
        try:
            current = file_identity(identity["path"])
        except OSError as exc:
            raise RuntimeError(
                f"offline passive replay {label} became unavailable {stage}"
            ) from exc
        if current != identity:
            raise RuntimeError(f"offline passive replay {label} changed {stage}")


def require_training_output_distinct(metadata, output: str | Path) -> None:
    require_distinct_artifact_paths(
        selection=metadata.dataset_identity["path"],
        training_report=metadata.training_identity["path"],
        output=output,
    )
