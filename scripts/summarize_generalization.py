"""Summarize measured artifacts without promoting checkpoints or hiding failures."""

import json
from pathlib import Path

root = Path("artifacts/generalization-20260905")
report = json.loads((root / "training/evaluation.json").read_text())
bench = json.loads((root / "current-throughput.json").read_text())
lines = [
    "# Cross-environment pilot — 2026-09-05",
    "",
    "Implemented seeded utility-plant, windowless data-center and bounded forest scenes. Native camera resolutions are 64×48, 128×96, 256×192, 512×384 and 768×576. The recurrent local policy remains 64×48 RGB-D with a shared observation-only mission planner, scan/altitude supervision, and modeled odometry. This is behavior cloning, not end-to-end reinforcement learning.",
    "",
    "Two 22,656-parameter policies were trained with identical model seed and optimizer budget: plant-only (1,800 unique samples) and mixed indoor (3,600). Entire forest family excluded from training and validation. Checkpoints fixed before test access. One test seed per family means these results are a pilot, not a statistically supported generalization claim.",
    "",
    "| Held-out family | Classical | Plant-only policy | Mixed-indoor policy |",
    "|---|---:|---:|---:|",
]
for family in ("utility_plant", "data_center", "forest"):
    values = [
        report["results"][f"{arm}_test/{family}/100"]
        for arm in ("classical", "plant_only", "mixed_indoor")
    ]
    lines.append(
        "| "
        + family.replace("_", " ")
        + " | "
        + " | ".join(
            f"{round(v['coverage'] * 3)}/3 targets; {'collision' if v['collision'] else 'no collision'}"
            for v in values
        )
        + " |"
    )
lines += [
    "",
    "Mixed training did not improve the held-out results here. Both learned policies transferred partially to the unseen forest, while data-center coverage stayed below the classical baseline. Do not promote this checkpoint as generally autonomous.",
    "",
    "## Throughput",
    "",
    "| Stage | Measured rate |",
    "|---|---:|",
]
for row in bench["physics"]:
    lines.append(
        f"| Native physics, batch {row['batch']} | {row['units_per_s'] / 1e6:.2f} million substeps/s |"
    )
for row in bench["training"]:
    lines.append(
        f"| MPS optimizer, batch {row['batch']} × sequence 8 | {row['units_per_s']:,.0f} samples/s |"
    )
lines += [
    f"| Full 256×192 dust pipeline | {bench['end_to_end']['sensor_ticks_per_s']:.1f} camera steps/s |",
    "",
    "Native batching is measured in one process; this is not a multi-worker scaling benchmark. Optimizer samples are repeated training examples, not newly rendered data. Full 128×96 rollout rates are recorded per episode in evaluation.json; approximately 28 camera steps/s for the indoor collection. Baseline particle workload is 4,096 parcels versus 128 in the training layouts, so the speed difference is not attributable only to resolution.",
    "",
    "Native RGB-D + optics at a fixed, initially clear pose: about 255–348 frames/s at 128×96, 90–107 at 256×192, 24–30 at 512×384, and 12–14 at 768×576. See resolution-throughput.json for individual observations. These are single-pose short measurements; there is no demonstrated navigation benefit from higher resolution yet.",
    "",
    "## Boundaries and next work",
    "",
    "Procedural surfaces and analytic trees are research geometry, not photorealistic CAD assets. Forest shadows/collisions use conservative authored bounds and navigation remains in a bounded plot. Dust wake is reduced-order, depth/attitude ideal, and inspection targets use diagnostic colors. No hardware flight or real-world transfer was evaluated.",
    "",
    "The next experiment should improve teacher data coverage, gather corrective examples in failed data-center states, and expand held-out seeds. Keep the current frozen pilot as the baseline. Renderer/aerosol throughput and persistent airflow deserve optimization before scaling to large visual training batches.",
    "",
    "Validation: 54 focused tests passed, including seeded geometry identity and exact high-resolution sensor-to-policy downsampling; viewer production build passed. Actual replays and camera renders are under artifacts/generalization-20260905.",
]
stress = root / "training/stress-results.json"
if stress.exists():
    lines += ["", "## Additional fixed stress probes", ""]
    for case, row in json.loads(stress.read_text())["results"].items():
        lines.append(
            f"- {case}: {round(row['coverage'] * 3)}/3 targets; collision={row['collision']}; recovered={row['recovered']}; outage observed={row['outage_observed']}; minimum transmission={row['minimum_transmission']:.1%}."
        )
Path("docs/environment-generalization-20260905.md").write_text("\n".join(lines) + "\n")
(root / "summary.md").write_text("\n".join(lines) + "\n")
