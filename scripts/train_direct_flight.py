"""Train a direct flight controller before inspecting its closed-loop test courses."""

import json
from pathlib import Path
from flightrl.fleet.flight_policy.model import train

root = Path("artifacts/direct-flight-20260906")
root.mkdir(exist_ok=False)
(root / "plan.json").write_text(
    json.dumps(
        {
            "training_seed": 44,
            "validation_seed": 45,
            "test_seeds": list(range(200, 212)),
            "demo_seed": 200,
            "outputs": ["collective thrust", "roll rate", "pitch rate", "yaw rate"],
            "inputs": "relative 3-D waypoint, velocity, attitude, heading error",
            "scope": "FPV native six-DOF; high-level waypoints supplied; no camera perception or learned route discovery",
        },
        indent=2,
    )
)
(root / "training.json").write_text(json.dumps(train(root / "controller.pt"), indent=2))
