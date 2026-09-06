"""Separate preregistered visibility and link-recovery probes; no model selection."""

import json
from dataclasses import replace
from pathlib import Path
from flightrl.artifact_identity import sha256_file
from flightrl.inspection.environments import environment_scene
from flightrl.inspection_scene import compile_inspection_scene
from flightrl.inspection.rollout import run_mission
from flightrl.inspection.student import StudentPolicy

root = Path("artifacts/generalization-20260905/training")
checkpoint = root / "mixed_indoor.pt"
plan = {
    "seed": 200,
    "family": "utility_plant",
    "ticks": 1800,
    "sensor_size": [128, 96],
    "checkpoint_sha256": sha256_file(checkpoint),
    "cases": ["low_visibility", "link_recovery"],
    "scope": "Additional fixed stress probes, not training or checkpoint selection",
}
(root / "stress-plan.json").write_text(json.dumps(plan, indent=2))
results = {}
for case in plan["cases"]:
    scene = environment_scene("utility_plant", 200)
    if case == "low_visibility":
        profile = replace(
            scene.environment, settled_fraction=0.2, dust_extinction_per_m=0.45
        )
        scene = compile_inspection_scene(
            scene.scenario, scene.panels, scene.evaluator_ids, environment=profile
        )
    run = run_mission(
        scene,
        industrial=True,
        ticks=1800,
        seed=200,
        sensor_size=(128, 96),
        policy=StudentPolicy(checkpoint),
        link_loss=case == "link_recovery",
    )
    result = run[0]
    result["minimum_transmission"] = min(r["mean_transmission"] for r in run[1])
    result["outage_observed"] = any(not r["connected"] for r in run[1])
    results[case] = result
    (root / "stress-results.json").write_text(
        json.dumps({"plan": plan, "results": results}, indent=2)
    )
    print(
        case,
        result["coverage"],
        result["collision"],
        result["recovered"],
        result["outage_observed"],
        flush=True,
    )
