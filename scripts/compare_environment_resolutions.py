"""Same-pose resolution costs and native camera artifacts across scene families."""

import json
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image
from flightrl import _binding
from flightrl.inspection.environments import environment_scene
from flightrl.environment.simulation import EnvironmentSimulation

root = Path("artifacts/generalization-20260905")
root.mkdir(exist_ok=True)
rows = []
for family in ("utility_plant", "data_center", "forest"):
    scene = environment_scene(family, 101)
    p = np.array([[-2, -1.5, 1.5]], np.float32)
    q = np.array([[1, 0, 0, 0]], np.float32)
    for w, h in ((128, 96), (256, 192), (512, 384), (768, 576)):
        env = EnvironmentSimulation(101, scene, (w, h))
        rgb = np.zeros((1, h, w, 3), np.uint8)
        depth = np.zeros((1, h, w), np.float32)
        counts = np.zeros((1, len(scene.panels), 2), np.int32)

        def render():
            _binding.inspection_render(
                p,
                q,
                env.room,
                env.boxes,
                scene.panels,
                rgb,
                counts,
                depth,
                1,
                *env.render_buffers,
            )
            env.camera(rgb[0], depth[0], p[0], q[0])
            env.optics.apply(rgb[0])

        render()
        start = perf_counter()
        for _ in range(5):
            render()
        elapsed = perf_counter() - start
        Image.fromarray(rgb[0]).save(root / f"{family}-{w}.png")
        rows.append(
            {
                "family": family,
                "resolution": [w, h],
                "frames_per_s": 5 / elapsed,
                "frame_bytes": rgb.nbytes + depth.nbytes,
                "scene_sha256": scene.manifest["sha256"],
            }
        )
        print(family, w, round(5 / elapsed, 1), flush=True)
(root / "resolution-throughput.json").write_text(json.dumps(rows, indent=2))
