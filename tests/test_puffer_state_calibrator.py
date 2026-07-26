from __future__ import annotations

from pathlib import Path
import importlib.util

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_puffer_state_calibrator.py"
SPEC = importlib.util.spec_from_file_location("train_puffer_state_calibrator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
calibrator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibrator)


def test_weighted_action_mse_prioritizes_weighted_rows() -> None:
    prediction = torch.tensor([[1.0, 0.0], [0.1, 0.0]])
    target = torch.zeros(2, 2)

    unweighted = calibrator.weighted_action_mse(prediction, target, torch.ones(2))
    weighted = calibrator.weighted_action_mse(prediction, target, torch.tensor([3.0, 0.5]))

    assert weighted > unweighted
