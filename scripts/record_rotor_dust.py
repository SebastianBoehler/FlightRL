"""Record an unfiltered 30-second native closed-loop dust-bed demonstration."""

import json
from pathlib import Path
from dataclasses import replace
import torch
from flightrl.inspection.industrial import utility_plant
from flightrl.inspection.rollout import run_mission
from flightrl.inspection.student import StudentPolicy
from flightrl.inspection_scene import compile_inspection_scene
from flightrl.artifact_identity import sha256_file
from evaluate_inspection_demo import export

root = Path("artifacts/rotor-dust-20260905b")
root.mkdir(exist_ok=False)
torch.set_num_threads(2)
checkpoint = Path("artifacts/utility-plant-optics-training-20260905/selected.pt")
base = utility_plant(400, heavy_dust=True)
profile = replace(
    base.environment,
    dust_extinction_per_m=1.2,
    settled_fraction=0.98,
    wind_m_s=(0.03, 0, 0),
    turbulence_m_s=0.02,
)
scene = compile_inspection_scene(
    base.scenario, base.panels, base.evaluator_ids, environment=profile
)
run = run_mission(
    scene, industrial=True, ticks=300, policy=StudentPolicy(checkpoint), seed=400
)
entry = export(root, "rotor-dust", scene, run, sha256_file(checkpoint))
records = run[1]
report = {
    "result": run[0],
    "resuspensions": records[-1]["dust_resuspensions"],
    "minimum_transmission": min(r["mean_transmission"] for r in records),
    "scope": "30-second frozen-policy demonstration; reduced-order rotor wake and finite settled dust bed, not calibrated CFD.",
}
(root / "evaluation.json").write_text(json.dumps(report, indent=2))
old = Path("artifacts/environment-engine-20260905")
entries = json.loads((old / "index.json").read_text())["episodes"]
for item in old.iterdir():
    if item.name not in ("index.json", "evaluation.json"):
        (root / item.name).symlink_to(item.resolve())
(root / "index.json").write_text(
    json.dumps(
        {
            "episodes": [entry, *entries],
            "evaluation": {
                "summary": "Rotor dust bed · 30 s native flight · cyan airflow / orange wind acceleration · earlier recordings retained"
            },
        }
    )
)
print(json.dumps(report), flush=True)
