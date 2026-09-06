"""Frozen local three-drone pilot; no claims of calibrated DJI or detailed-mesh transfer."""

import json
from dataclasses import asdict
from pathlib import Path
from flightrl.inspection.environments import environment_scene
from flightrl.fleet.rollout import run_fleet
from flightrl.fleet.student import train, FleetPolicy
from flightrl.fleet.vehicles import VEHICLES
from flightrl.artifact_identity import sha256_file

root = Path("artifacts/fleet-pilot-20260905-v2")
root.mkdir(exist_ok=False)
plan = {
    "train_families": ["utility_plant", "data_center"],
    "train_vehicles": ["fpv", "industrial"],
    "train_missions": ["inspect", "delivery"],
    "train_seeds": [20],
    "test_seeds": [120, 121],
    "held_out_family": "forest",
    "held_out_vehicle": "agriculture",
    "held_out_mission": "return",
    "drones": 3,
    "ticks": 200,
    "epochs": 12,
    "sensor_resolution": [64, 48],
    "communication": "5 Hz, 200 ms latency, 1 s TTL; estimated pose/velocity, assignment, completion; no images",
    "limits": "Analytic native scenes, not detailed WebGPU geometry; assumed motor/rate response; no inter-drone wake or battery model",
    "vehicle_references": {k: asdict(v) for k, v in VEHICLES.items()},
}
(root / "plan.json").write_text(json.dumps(plan, indent=2))
data = []
results = {}
for family in plan["train_families"]:
    for vehicle in plan["train_vehicles"]:
        for mission in plan["train_missions"]:
            key = f"train/{family}/{vehicle}/{mission}/20"
            metrics, records, samples = run_fleet(
                environment_scene(family, 20), vehicle, mission, 20, collect=True
            )
            results[key] = metrics
            data.extend(samples)
            print(key, metrics["status"], len(samples), flush=True)
if not data:
    raise RuntimeError("no training samples")
checkpoint = root / "fleet.pt"
training = train(data, checkpoint, plan["epochs"])
(root / "checkpoint.json").write_text(
    json.dumps({**training, "sha256": sha256_file(checkpoint)}, indent=2)
)
policy = FleetPolicy(checkpoint)
for family in (*plan["train_families"], "forest"):
    for vehicle in VEHICLES:
        for mission in ("inspect", "delivery", "return"):
            for seed in plan["test_seeds"]:
                key = f"test/{family}/{vehicle}/{mission}/{seed}"
                metrics, records, _ = run_fleet(
                    environment_scene(family, seed),
                    vehicle,
                    mission,
                    seed,
                    policy=policy,
                )
                results[key] = metrics
                if (
                    family == "forest"
                    and vehicle == "fpv"
                    and mission == "inspect"
                    and seed == 120
                ):
                    (root / "replay.json").write_text(
                        json.dumps({"records": records, "result": metrics})
                    )
                print(key, metrics["status"], metrics.get("coverage"), flush=True)
                (root / "results.json").write_text(json.dumps(results, indent=2))
(root / "results.json").write_text(json.dumps(results, indent=2))
