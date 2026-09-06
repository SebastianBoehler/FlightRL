"""Apply predeclared quality checks to retained evaluation results."""

import json
from pathlib import Path
import numpy as np

root = Path("artifacts/reconstruction-20260906/heldout-repaired")
results = json.loads((root / "evaluation.json").read_text())
thresholds = json.loads((root / "plan.json").read_text())["thresholds"]
summary = {
    "episodes": len(results),
    "mission_complete": sum(r["mission"]["status"] == "complete" for r in results),
    "collisions": sum(r["mission"]["status"] == "collision" for r in results),
    "incorrect_reports": sum(
        r["mission"]["status"] == "incorrect_report" for r in results
    ),
    "camera_frames_per_s": sum(r["camera_frames"] for r in results)
    / sum(r["reconstruction_wall_s"] for r in results),
    "quality_thresholds": thresholds,
    "mapping": {},
}
for mode in ("rgb", "rgbd"):
    maps = [m for r in results for m in r["mapping"] if m["mode"] == mode]
    passed = [
        m
        for m in maps
        if m["tracking_fraction"] >= thresholds["tracking_fraction"]
        and m["ate_rmse_m"] is not None
        and m["ate_rmse_m"] <= thresholds["ate_rmse_m"]
    ]
    summary["mapping"][mode] = dict(
        sequences=len(maps),
        passed_tracking_and_drift_checks=len(passed),
        mean_tracking_fraction=float(np.mean([m["tracking_fraction"] for m in maps])),
        median_ate_rmse_m=float(
            np.median([m["ate_rmse_m"] for m in maps if m["ate_rmse_m"] is not None])
        ),
        median_surface_accuracy_m=float(
            np.median(
                [
                    m["surface_accuracy_m"]
                    for m in maps
                    if m["surface_accuracy_m"] is not None
                ]
            )
        ),
    )
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
