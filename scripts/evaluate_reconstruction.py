"""Frozen camera actor + independent reconstruction evaluation, never overwrite runs."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
from PIL import Image
from flightrl.reconstruction.experiment import experiment
from flightrl.fleet.camera_policy.network import Policy

p = argparse.ArgumentParser()
p.add_argument("output", type=Path)
p.add_argument("--seeds", type=int, nargs="+", required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
actor = Path("artifacts/camera-control-linkloss-20260906/actor.pt")
plan = dict(
    seeds=a.seeds,
    actor_sha256=hashlib.sha256(actor.read_bytes()).hexdigest(),
    backend="Incremental visual odometry, no loop closure",
    hardware=platform.platform(),
    thresholds=dict(surface_distance_m=0.15, tracking_fraction=0.95, ate_rmse_m=0.15),
    source_hashes={
        str(f): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in Path("src/flightrl/reconstruction").glob("*.py")
    },
)
(a.output / "plan.json").write_text(json.dumps(plan, indent=2))
policy = Policy(actor)
results = []
for seed in a.seeds:
    result, review, frames = experiment(seed, policy)
    results.append(result)
    (a.output / f"{seed}.json").write_text(json.dumps(result, indent=2))
    if seed == a.seeds[0]:
        (a.output / "review.json").write_text(json.dumps(review, separators=(",", ":")))
        # Browser-safe paged atlases; no very tall images exceeding texture limits.
        for page, start in enumerate(range(0, len(frames), 32)):
            Image.fromarray(np.concatenate(frames[start : start + 32], axis=0)).save(
                a.output / f"camera-{page}.jpg", quality=88
            )
        for entry in review["maps"]:
            rows = entry["points"]
            lines = [
                "ply",
                "format ascii 1.0",
                f"element vertex {len(rows)}",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
            ]
            lines += [" ".join(map(str, point + color)) for point, color, _ in rows]
            (a.output / f"drone-{entry['drone']}-{entry['mode']}.ply").write_text(
                "\n".join(lines) + "\n"
            )
    print(
        seed,
        result["mission"]["status"],
        [
            (m["mode"], round(m["tracking_fraction"], 2), m["ate_rmse_m"])
            for m in result["mapping"]
        ],
        flush=True,
    )
(a.output / "evaluation.json").write_text(json.dumps(results, indent=2))
