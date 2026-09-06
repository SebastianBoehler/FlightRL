"""Frozen-policy regression of own-engine atmosphere; no success filtering."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from flightrl.inspection.industrial import utility_plant
from flightrl.inspection.rollout import run_mission
from flightrl.inspection.student import StudentPolicy
from flightrl.inspection_scene import compile_inspection_scene
from flightrl.artifact_identity import sha256_file
from evaluate_inspection_demo import export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    actor = StudentPolicy(args.checkpoint)
    identity = sha256_file(args.checkpoint)
    base = utility_plant(400)
    profiles = [
        base.environment,
        replace(base.environment, name="heavy_dust", dust_extinction_per_m=0.45),
        replace(
            base.environment,
            name="gusty_plant",
            wind_m_s=(0.7, -0.25, 0.05),
            turbulence_m_s=0.4,
        ),
    ]
    (args.output / "frozen-plan.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": identity,
                "seed": 400,
                "ticks": 1800,
                "profiles": [p.report() for p in profiles],
            },
            indent=2,
        )
    )
    results = {}
    episodes = []
    for profile in profiles:
        scene = compile_inspection_scene(
            base.scenario, base.panels, base.evaluator_ids, environment=profile
        )
        run = run_mission(
            scene, industrial=True, ticks=1800, policy=actor, seed=400, collect=True
        )
        result, records, frames, depth, samples = run
        # Store actual training-ready samples, preserving episode reset boundary.
        np.savez_compressed(
            args.output / f"{profile.name}-training.npz",
            rgb=np.stack([s[0] for s in samples]),
            depth=np.stack([s[1] for s in samples]),
            proprio=np.stack([s[2] for s in samples]),
            teacher=np.stack([s[3] for s in samples]),
        )
        result["mean_camera_transmission"] = float(
            np.mean([r["mean_transmission"] for r in records])
        )
        result["max_resuspensions"] = max(r["dust_resuspensions"] for r in records)
        result["final_deposited"] = records[-1]["dust_deposited"]
        results[profile.name] = result
        episodes.append(
            export(args.output, profile.name.replace("_", "-"), scene, run, identity)
        )
        print(
            profile.name,
            result["status"],
            result["coverage"],
            result["mean_camera_transmission"],
            flush=True,
        )
    report = {
        "results": results,
        "checkpoint_sha256": identity,
        "retrained": False,
        "scope": "One fixed policy and one authored layout across three specified atmospheric profiles. Regression evidence, not a generalization benchmark.",
        "sensor": "Scene-owned materials/lights; spatial aerosol extinction and shadowed scattering; Metal lens pass; ideal depth; actual 256x192 RGB averaged to 64x48 training input.",
        "physics": "Reduced-order coherent airflow, air-relative native drag, four rotor wakes, inertial parcels with settling, swept contacts and floor resuspension. No CFD, particle erosion, motor contamination or calibrated aerosol claim.",
    }
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2))
    summary = "Own-engine atmosphere · 3 recorded conditions · frozen policy · see per-run results"
    (args.output / "index.json").write_text(
        json.dumps({"episodes": episodes, "evaluation": {"summary": summary}}, indent=2)
    )


if __name__ == "__main__":
    main()
