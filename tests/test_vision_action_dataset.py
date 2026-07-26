from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from flightrl.vision import VisionActionScale, load_aligned_vision_actions, phase_holdout_split


def test_load_aligned_vision_actions_and_phase_split(tmp_path: Path) -> None:
    vision = tmp_path / "vision.npz"
    telemetry = tmp_path / "telemetry.csv"
    np.savez_compressed(
        vision,
        observations=np.zeros((6, 3, 4, 5), dtype=np.float32),
        host_time_s=np.arange(6, dtype=np.float64) * 0.02 + 10.0,
        contract_json=np.asarray(json.dumps({"shape": [3, 4, 5]})),
    )
    fieldnames = [
        "host_time_s",
        "sys.isFlying",
        "sys.isTumbled",
        "baseline_controls_drone",
        "baseline_vx_m_s",
        "baseline_vy_m_s",
        "baseline_yawrate_deg_s",
        "phase",
    ]
    with telemetry.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(6):
            writer.writerow(
                {
                    "host_time_s": 10.0 + index * 0.02,
                    "sys.isFlying": 1,
                    "sys.isTumbled": 0,
                    "baseline_controls_drone": True,
                    "baseline_vx_m_s": 0.15 if index < 3 else 0.0,
                    "baseline_vy_m_s": 0.0,
                    "baseline_yawrate_deg_s": 0.0 if index < 3 else 60.0,
                    "phase": "line" if index < 3 else "yaw",
                }
            )

    dataset = load_aligned_vision_actions([vision], [telemetry], scale=VisionActionScale())
    train, validation = phase_holdout_split(dataset, validation_fraction=0.34)

    assert dataset.observations.shape == (6, 3, 4, 5)
    np.testing.assert_allclose(dataset.actions[:3, 0], 1.0)
    np.testing.assert_allclose(dataset.actions[3:, 2], 1.0)
    assert len(train) == 4
    assert len(validation) == 2
