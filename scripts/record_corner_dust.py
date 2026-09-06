"""Native scripted corner experiment: approach, loft dust, retreat, settle."""

import json
from dataclasses import replace
from pathlib import Path
from flightrl.inspection.industrial import utility_plant
from flightrl.inspection.dust_demo import CornerDustDemo
from flightrl.inspection.rollout import run_mission
from flightrl.inspection_scene import compile_inspection_scene
from evaluate_inspection_demo import export

root = Path("artifacts/corner-dust-20260905-contact-fix")
root.mkdir(exist_ok=False)
base = utility_plant(400)
profile = replace(
    base.environment,
    name="corner_dust",
    particle_count=8192,
    settled_fraction=1.0,
    dust_bed_bounds=(-3.8, -2.65, -2.9, -1.85),
    wind_m_s=(0.02, 0, 0),
    turbulence_m_s=0.015,
    dust_extinction_per_m=0.025,
    grain_diameter_um=(20, 60),
)
scene = compile_inspection_scene(
    base.scenario, base.panels, base.evaluator_ids, environment=profile
)
run = run_mission(
    scene, industrial=True, ticks=600, seed=400, controller_factory=CornerDustDemo
)
entry = export(root, "corner-dust", scene, run)
(root / "evaluation.json").write_text(
    json.dumps(
        {
            "result": run[0],
            "scope": "Scripted native dynamics experiment, not autonomous navigation or calibrated CFD.",
            "physics": "Finite settled bed; spherical grains 20–60 um, gravity/buoyancy and Reynolds-corrected drag; approximate rotor airflow.",
            "transmission": [
                (r["time_s"], r["mean_transmission"], r["dust_airborne"])
                for r in run[1][::50]
            ],
        },
        indent=2,
    )
)
(root / "index.json").write_text(
    json.dumps(
        {
            "episodes": [entry],
            "evaluation": {
                "summary": "Corner dust · scripted native flight · 20–60 μm grains · approach / stir / retreat / settle"
            },
        }
    )
)
print((root / "evaluation.json").read_text(), flush=True)
