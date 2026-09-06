"""Fresh utility-plant evaluation; recorded sensor pixels are also training inputs."""

import argparse, json
from pathlib import Path
import torch
from flightrl.artifact_identity import sha256_file
from flightrl.inspection.industrial import utility_plant
from flightrl.inspection.rollout import run_mission
from flightrl.inspection.student import StudentPolicy
from evaluate_inspection_demo import export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    policy = StudentPolicy(args.checkpoint)
    identity = sha256_file(args.checkpoint)
    report = {
        "checkpoint_sha256": identity,
        "test_seeds": [300, 301, 302, 303],
        "results": {},
        "scope": "Authored utility plant; equipment shift and independently seeded gust/dust variations. Four held-out seeds, not generalization to arbitrary buildings.",
        "sensor": "256x192 procedural material and window/direct lighting, glossy highlights, Metal bloom/streak/vignette/grain lens pass with 0.035/m aerosol extinction and advected particles; exact 4x4 average to 64x48 policy RGB; ideal ray depth; modeled velocity odometry, ideal attitude.",
        "physics": "Native 6DOF with OU acceleration: tau 0.6s, stationary std [0.10,0.10,0.025] m/s2. Conservative box obstacles and 0.08m swept body margin. Not CFD or calibrated optics.",
    }
    (args.output / "frozen-plan.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": identity,
                "test_seeds": report["test_seeds"],
                "ticks": 1800,
                "gate": "all panels, no collisions, link recovery",
            },
            indent=2,
        )
    )
    episodes = []
    for name, actor in [("classical", None), ("learned", policy)]:
        rows = []
        for seed in report["test_seeds"]:
            scene = utility_plant(seed)
            run = run_mission(
                scene, ticks=1800, industrial=True, policy=actor, seed=seed
            )
            rows.append(run[0])
            print(name, seed, run[0]["coverage"], run[0]["status"], flush=True)
            if seed == 300 or (run[0]["coverage"] < 1 or run[0]["collision"]):
                episodes.append(
                    export(
                        args.output,
                        name
                        + (f"-plant-incomplete-{seed}" if seed != 300 else "-plant"),
                        scene,
                        run,
                        identity if actor else None,
                    )
                )
        report["results"][name] = rows
    scene = utility_plant(300)
    recovery = run_mission(
        scene, ticks=1800, industrial=True, policy=policy, seed=300, link_loss=True
    )
    report["recovery"] = recovery[0]
    episodes.append(
        export(args.output, "plant-link-recovery", scene, recovery, identity)
    )
    report["student_promoted"] = (
        all(
            r["coverage"] == 1 and not r["collision"]
            for r in report["results"]["learned"]
        )
        and recovery[0]["recovered"]
    )
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2))
    average = lambda name: sum(r["coverage"] for r in report["results"][name]) / 4
    summary = f"Utility plant · 4 held-out seeds · Classical {average('classical'):.0%} · Learned {average('learned'):.0%} coverage · {'Gate passed' if report['student_promoted'] else 'Gate not passed'}"
    episodes.sort(key=lambda e: 0 if e["file"] == "learned-plant.json" else 1)
    (args.output / "index.json").write_text(
        json.dumps({"episodes": episodes, "evaluation": {"summary": summary}}, indent=2)
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
