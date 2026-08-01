from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from flightrl.evidence_scope import file_identity
from flightrl.puffer4_edge_checkpoint import require_edge_checkpoint
from puffer4_edge_artifact_support import checkpoint_artifacts


def test_checkpoint_rejects_report_with_uneven_tbptt_chunks(tmp_path: Path) -> None:
    artifacts = checkpoint_artifacts(tmp_path)
    payload = torch.load(
        artifacts.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    training = Path(payload["training_identity"]["path"])
    report = json.loads(training.read_text())
    report["config"]["tbptt_steps"] = 3
    training.write_text(json.dumps(report) + "\n")
    payload["training_identity"] = file_identity(training)

    with pytest.raises(ValueError, match="divide evenly"):
        require_edge_checkpoint(payload)
