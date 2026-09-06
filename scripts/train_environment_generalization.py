"""Predeclared small cross-family pilot; no test-based checkpoint selection."""

import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from flightrl.inspection.environments import environment_scene
from flightrl.inspection.rollout import run_mission
from flightrl.inspection.student import StudentPolicy
from flightrl.artifact_identity import sha256_file
from train_inspection_student import train
from evaluate_inspection_demo import export

root = Path("artifacts/generalization-20260905/training")
root.mkdir(exist_ok=False)
torch.set_num_threads(2)
plan = {
    "train": {"utility_plant": [10, 11], "data_center": [10, 11]},
    "validation": {"utility_plant": [50], "data_center": [50]},
    "test": {"utility_plant": [100], "data_center": [100], "forest": [100]},
    "training_ticks": 900,
    "validation_ticks": 600,
    "test_ticks": 1200,
    "sensor_size": [128, 96],
    "policy_input": [64, 48],
    "model_seed": 0,
    "epochs": 20,
    "arms": ["plant_only", "mixed_indoor"],
    "selection": "fixed final checkpoint; no test selection",
    "task": "RGB-D marker inspection with shared observation-only planner and safety supervisor",
    "forest": "entire family excluded from training and validation",
    "scope": "Local pilot with one held-out test seed per family, not statistical proof of generalization",
}
(root / "frozen-plan.json").write_text(json.dumps(plan, indent=2))
records = {}
datasets = {}
episodes = []


def rollout(family, seed, ticks, policy=None, collect=False):
    return run_mission(
        environment_scene(family, seed),
        ticks=ticks,
        seed=seed,
        policy=policy,
        collect=collect,
        industrial=True,
        sensor_size=(128, 96),
    )


def metrics(run):
    result, frames = run[0], run[1]
    return {
        **result,
        "completion": result["coverage"] == 1 and not result["collision"],
        "minimum_camera_clearance_m": min(f["clearance"] for f in frames),
        "mean_transmission": float(np.mean([f["mean_transmission"] for f in frames])),
        "sensor_steps_per_s": result["ticks"] / result["wall_s"],
    }


for family, seeds in plan["train"].items():
    datasets[family] = []
    for seed in seeds:
        run = rollout(family, seed, plan["training_ticks"], collect=True)
        datasets[family].extend((*sample, i == 0) for i, sample in enumerate(run[-1]))
        records[f"teacher_train/{family}/{seed}"] = metrics(run)
        print("collected", family, seed, len(run[-1]), run[0]["coverage"], flush=True)
    samples = datasets[family]
    np.savez_compressed(
        root / f"{family}-training.npz",
        rgb=np.stack([s[0] for s in samples]),
        depth=np.stack([s[1] for s in samples]),
        proprio=np.stack([s[2] for s in samples]),
        teacher=np.stack([s[3] for s in samples]),
        reset=np.array([s[4] for s in samples]),
    )

training = {}
for arm in plan["arms"]:
    data = datasets["utility_plant"] + (
        datasets["data_center"] if arm == "mixed_indoor" else []
    )
    path = root / f"{arm}.pt"
    start = perf_counter()
    training[arm] = train(data, 0, path, epochs=plan["epochs"])
    training[arm].update(
        wall_s=perf_counter() - start,
        unique_samples=len(data),
        checkpoint_sha256=sha256_file(path),
    )
    for family, seeds in plan["validation"].items():
        for seed in seeds:
            run = rollout(family, seed, plan["validation_ticks"], StudentPolicy(path))
            records[f"{arm}_validation/{family}/{seed}"] = metrics(run)
            print(
                "validation",
                arm,
                family,
                run[0]["coverage"],
                run[0]["collision"],
                flush=True,
            )
    (root / "progress.json").write_text(
        json.dumps({"training": training, "results": records}, indent=2)
    )

# Freeze both model hashes before opening the test split.
(root / "frozen-checkpoints.json").write_text(json.dumps(training, indent=2))
for family, seeds in plan["test"].items():
    for seed in seeds:
        for arm in ["classical", *plan["arms"]]:
            path = None if arm == "classical" else root / f"{arm}.pt"
            run = rollout(
                family,
                seed,
                plan["test_ticks"],
                None if path is None else StudentPolicy(path),
            )
            records[f"{arm}_test/{family}/{seed}"] = metrics(run)
            if arm == "mixed_indoor":
                episodes.append(
                    export(
                        root,
                        f"{family}-held-out",
                        environment_scene(family, seed),
                        run,
                        sha256_file(path),
                    )
                )
            print(
                "test", arm, family, run[0]["coverage"], run[0]["collision"], flush=True
            )
            (root / "progress.json").write_text(
                json.dumps({"training": training, "results": records}, indent=2)
            )
report = {
    "plan": plan,
    "training": training,
    "results": records,
    "limitations": [
        "Single test seed per family",
        "Ideal depth and attitude",
        "Shared classical planner, scan and altitude supervision",
        "Diagnostic colored inspection targets",
        "Procedural materials and conservative collision bounds",
        "No real-world transfer claim",
    ],
}
(root / "evaluation.json").write_text(json.dumps(report, indent=2))
(root / "index.json").write_text(
    json.dumps(
        {
            "episodes": episodes,
            "evaluation": {
                "summary": "Held-out layouts · mixed indoor visual policy · forest excluded from training · see measured results"
            },
        }
    )
)
print("completed", root, flush=True)
