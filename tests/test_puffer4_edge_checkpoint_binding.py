from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from flightrl.evidence_scope import file_identity
from flightrl.puffer4_edge_checkpoint import require_edge_checkpoint
from flightrl.puffer4_edge_training_state import edge_state_dict_sha256
from puffer4_edge_artifact_support import checkpoint_artifacts


def _payload(tmp_path: Path) -> dict:
    artifacts = checkpoint_artifacts(tmp_path)
    return torch.load(artifacts.checkpoint, map_location="cpu", weights_only=True)


def test_edge_checkpoint_binds_embedded_actor_state_to_training_report(
    tmp_path: Path,
) -> None:
    payload = _payload(tmp_path)
    payload["state_dict"]["action_head.0.bias"].reshape(-1)[0] += 0.1

    with pytest.raises(ValueError, match="state digest"):
        require_edge_checkpoint(payload)


def test_edge_checkpoint_recomputes_selected_metrics_from_embedded_actor(
    tmp_path: Path,
) -> None:
    payload = _payload(tmp_path)
    payload["state_dict"]["action_head.0.bias"].reshape(-1)[0] += 0.1
    training = Path(payload["training_identity"]["path"])
    report = json.loads(training.read_text())
    report["selected_actor_state_sha256"] = edge_state_dict_sha256(
        payload["state_dict"]
    )
    _rebind_training(payload, training, report)

    with pytest.raises(ValueError, match="clean metrics do not reproduce"):
        require_edge_checkpoint(payload)


def test_edge_checkpoint_recomputes_visual_ablation_metrics(
    tmp_path: Path,
) -> None:
    payload = _payload(tmp_path)
    training = Path(payload["training_identity"]["path"])
    report = json.loads(training.read_text())
    index = report["best_epoch"] - 1
    forged = report["history"][index]["selection_visual_ablation"]
    forged["decision_action_loss"] += 0.01
    forged["total_loss"] += 0.01
    forged["selection_score"] += 0.01
    report["best_selection_visual_ablation_metrics"] = dict(forged)
    _rebind_training(payload, training, report)

    with pytest.raises(ValueError, match="ablation metrics do not reproduce"):
        require_edge_checkpoint(payload)


def _rebind_training(payload: dict, path: Path, report: dict) -> None:
    path.write_text(json.dumps(report) + "\n")
    payload["training_identity"] = file_identity(path)
