"""Score the learned trial after inference, with evaluation-only references."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
from flightrl.fleet.camera_policy.network import Policy
from flightrl.reconstruction.experiment import experiment
from flightrl.reconstruction.metrics import score

folder = Path("artifacts/reconstruction-20260906/learned-trial")
pred = np.load(folder / "predictions.npz")
runtime = json.loads((folder / "runtime.json").read_text())
reference = []
result, review, rendered = experiment(
    4000,
    Policy("artifacts/camera-control-linkloss-20260906/actor.pt"),
    reference_out=reference,
)
indices = pred["indices"]
np.testing.assert_array_equal(
    np.stack([rendered[i][:, :256] for i in indices]), np.load(folder / "input-rgb.npy")
)
truth = np.tile(np.eye(4), (len(indices), 1, 1))
truth[:, :3, 3] = np.array(review["maps"][0]["truth"])[indices]
surface = []  # Image-space decimation avoids assuming a metric voxel size for monocular output.
for i, (points, confidence, colors) in enumerate(
    zip(pred["points"], pred["confidence"], pred["colors"])
):
    pts = points[::8, ::8].reshape(-1, 3)
    conf = confidence[::8, ::8].ravel()
    rgb = (colors[::8, ::8].reshape(-1, 3) * 255).astype(np.uint8)
    valid = np.isfinite(pts).all(1) & np.isfinite(conf) & (conf >= 1.5)
    # Scale warmup is bidirectional: do not reveal these points before all four captures.
    surface.extend(
        (p.tolist(), c.tolist(), max(i + 1, runtime["num_scale_frames"]))
        for p, c in zip(pts[valid], rgb[valid])
    )
metrics = score(list(pred["poses"]), truth, surface, reference[0], "rgb")
report = dict(
    runtime=runtime,
    metrics=metrics,
    points=len(surface),
    mission=result["mission"],
    confidence_threshold=1.5,
    limitations="One development trajectory, RGB only, 1 Hz keyframes, 4-frame initialization, evaluator-only Sim3. No relocalization or active exploration.",
)
(folder / "evaluation.json").write_text(json.dumps(report, indent=2))
entry = dict(
    drone=1,
    mode="rgb",
    points=surface,
    poses=[
        None if i < runtime["num_scale_frames"] - 1 else p[:3, 3].tolist()
        for i, p in enumerate(pred["poses"])
    ],
    truth=truth[:, :3, 3].tolist(),
    metrics=metrics,
)
output = dict(
    seed=4000,
    frames=len(indices),
    dt=1.0,
    maps=[entry],
    result=dict(mission=result["mission"], registration=[]),
    backend="LingBot-Map · published weights · RGB only",
    warmup=4,
)
(folder / "review.json").write_text(json.dumps(output, separators=(",", ":")))
input_rgb = np.load(folder / "input-rgb.npy")
for page, start in enumerate(range(0, len(input_rgb), 32)):
    Image.fromarray(
        np.concatenate(
            [
                np.concatenate([im, im, im], axis=1)
                for im in input_rgb[start : start + 32]
            ],
            axis=0,
        )
    ).save(folder / f"camera-{page}.jpg", quality=88)
rows = surface
header = [
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
(folder / "drone-1-rgb.ply").write_text(
    "\n".join(header + [" ".join(map(str, p + c)) for p, c, _ in rows]) + "\n"
)
print(json.dumps(report, indent=2))
