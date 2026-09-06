"""Freeze one checkpoint, compare 20 held-out layouts, export actual episode replays."""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from flightrl.artifact_identity import bind_payload, sha256_file
from flightrl.inspection.scenarios import scenario, SPLITS, GATES
from flightrl.inspection.student import StudentPolicy
from flightrl.inspection.rollout import run_mission
from flightrl.inspection_fixture import three_panel_room
from flightrl.inspection_replay_io import write_scene


def export(root, name, scene, run, policy_hash=None):
    result, records, frames, depth, _ = run
    h, w = frames.shape[1:3]
    atlas = Image.new("RGB", (20 * w, ((len(frames) + 19) // 20) * h))
    for i, frame in enumerate(frames):
        atlas.paste(Image.fromarray(frame), ((i % 20) * w, (i // 20) * h))
    atlas.save(root / f"{name}-frames.png")
    np.savez_compressed(root / f"{name}-sensor-recording.npz", rgb=frames, depth=depth)
    episode = {
        "name": name,
        "scene": {
            "room": scene.scenario.arrays["terrain_bounds"].tolist(),
            "boxes": scene.scenario.arrays["terrain_obstacles"].tolist(),
            "panels": scene.panels.tolist(),
            "identity": scene.manifest["sha256"],
            "environment": scene.environment.report() if scene.environment is not None else None,
        },
        "records": records,
        "result": result,
        "atlas": f"{name}-frames.png",
        "atlasColumns": 20,
        "frameWidth": w,
        "frameHeight": h,
        "policyHash": policy_hash,
        "sensor": "RGB-D camera; rendering and optical assumptions are documented in evaluation.json",
        "estimate": "known takeoff origin plus integrated noisy biased velocity; ideal attitude",
        "link_model": "spatial loss at x>=1.15; reconnect x<0.65; actor observes only boolean status",
        "atlas_sha256": sha256_file(root / f"{name}-frames.png"),
        "sensor_sha256": sha256_file(root / f"{name}-sensor-recording.npz"),
    }
    (root / f"{name}.json").write_text(
        json.dumps(bind_payload(episode), separators=(",", ":"))
    )
    return {"name": name.replace("-", " ").title(), "file": f"{name}.json"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    torch.set_num_threads(2)
    checkpoint_hash = sha256_file(args.checkpoint)
    policy = StudentPolicy(args.checkpoint)
    (args.output / "frozen-plan.json").write_text(
        json.dumps(
            {"splits": SPLITS, "gates": GATES, "checkpoint_sha256": checkpoint_hash},
            indent=2,
        )
    )
    results = {}
    for label, actor in (("classical", None), ("student", policy)):
        rows = []
        for seed in SPLITS["test"]:
            scene = scenario(seed)
            normal = run_mission(
                scene, ticks=GATES["mission_ticks"], policy=actor, seed=seed
            )[0]
            recovery = run_mission(
                scene,
                ticks=GATES["mission_ticks"],
                policy=actor,
                seed=seed,
                link_loss=True,
            )[0]
            rows.append({"seed": seed, "inspection": normal, "recovery": recovery})
            print(
                label,
                seed,
                normal["coverage"],
                normal["collision"],
                recovery["status"],
                flush=True,
            )
        results[label] = {
            "layouts": rows,
            "mean_coverage": float(
                np.mean([r["inspection"]["coverage"] for r in rows])
            ),
            "collision_rate": float(
                np.mean(
                    [
                        r["inspection"]["collision"] or r["recovery"]["collision"]
                        for r in rows
                    ]
                )
            ),
            "recovery_rate": float(np.mean([r["recovery"]["recovered"] for r in rows])),
        }
    failures = {}
    for fault in ("blocked_route", "continued_outage", "estimator_loss"):
        failures[fault] = run_mission(
            three_panel_room(), ticks=900, policy=policy, link_loss=True, fault=fault
        )[0]
    student = results["student"]
    promoted = (
        student["mean_coverage"] >= 0.9
        and student["collision_rate"] == 0
        and student["recovery_rate"] >= 0.9
    )
    report = {
        "checkpoint_sha256": checkpoint_hash,
        "gates": GATES,
        "splits": SPLITS,
        "results": results,
        "failure_injections": failures,
        "student_promoted": promoted,
        "wall_s": time.perf_counter() - start,
        "scope": "narrow held-out authored room variations; ideal RGB-D; modeled odometry; no real-flight generalization",
        "ppo_comparison": "not run; same-information classical comparison and 3 distillation seeds implemented",
    }
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2))
    scene = three_panel_room()
    write_scene(scene, args.output / "scene")
    episodes = []
    for name, actor, loss in (
        ("classical-inspection", None, False),
        ("learned-inspection", policy, False),
        ("link-recovery", policy, True),
    ):
        episodes.append(
            export(
                args.output,
                name,
                scene,
                run_mission(scene, ticks=900, policy=actor, link_loss=loss),
                checkpoint_hash if actor else None,
            )
        )
    summary = f"20 held-out layouts · Classical {results['classical']['mean_coverage']:.0%} coverage · Learned {student['mean_coverage']:.0%} · {'Gate passed' if promoted else 'Gate not passed'}"
    index = {
        "episodes": episodes,
        "evaluation": {
            "classicalCoverage": results["classical"]["mean_coverage"],
            "studentCoverage": student["mean_coverage"],
            "studentPromoted": promoted,
            "testLayouts": 20,
            "summary": summary,
        },
    }
    (args.output / "index.json").write_text(json.dumps(index, indent=2))
    assert sha256_file(args.checkpoint) == checkpoint_hash
    print(summary, flush=True)


if __name__ == "__main__":
    main()
