"""Stage timings on real scene/sensor data; batch size is not a thread count."""

import json
import platform
import resource
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from flightrl import _binding
from flightrl.inspection.industrial import utility_plant
from flightrl.inspection.rollout import run_mission
from flightrl.inspection.student import VisualController, image_tensor
from flightrl.environment.simulation import EnvironmentSimulation
from flightrl.sixdof.native import native_step


def measure(fn, count, units=1, sync=lambda: None):
    for _ in range(3):
        fn()
    sync()
    start = perf_counter()
    for _ in range(count):
        fn()
    sync()
    elapsed = perf_counter() - start
    return {
        "wall_s": elapsed,
        "iterations": count,
        "units_per_iteration": units,
        "units_per_s": count * units / elapsed,
    }


def main():
    torch.set_num_threads(2)
    scene = utility_plant(400)
    room, boxes = (
        scene.scenario.arrays[k] for k in ("terrain_bounds", "terrain_obstacles")
    )
    report = {
        "hardware": platform.platform(),
        "torch": torch.__version__,
        "cpu_threads": 2,
        "scene_sha256": scene.manifest["sha256"],
        "physics": [],
        "render": [],
        "training": [],
    }
    for n in (1, 8, 32):
        p = np.tile(np.array([-2, -1.5, 1.5], np.float32), (n, 1))
        v = np.zeros_like(p)
        q = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))
        rates = v.copy()
        ranges = np.empty((n, 6), np.float32)
        action = np.zeros((n, 4), np.float32)
        thrust = np.ones(n, np.float32)
        physics = np.repeat(scene.scenario.arrays["vehicle_physics"][None], n, axis=0)
        report["physics"].append(
            {
                "batch": n,
                **measure(
                    lambda: native_step(
                        p, v, q, rates, ranges, action, 0.02, room, thrust, physics
                    ),
                    100000,
                    n,
                ),
            }
        )
        p[:] = [-2, -1.5, 1.5]
        q[:] = [1, 0, 0, 0]
        for w, h in ((64, 48), (256, 192)):
            rgb = np.zeros((n, h, w, 3), np.uint8)
            depth = np.zeros((n, h, w), np.float32)
            counts = np.zeros((n, len(scene.panels), 2), np.int32)
            report["render"].append(
                {
                    "batch": n,
                    "resolution": [w, h],
                    **measure(
                        lambda: _binding.inspection_render(
                            p,
                            q,
                            room,
                            boxes,
                            scene.panels,
                            rgb,
                            counts,
                            depth,
                            1,
                            *scene.environment.render_buffers(),
                        ),
                        5,
                        n,
                    ),
                }
            )
    env = EnvironmentSimulation(400, scene)
    velocity = np.zeros((1, 3), np.float32)
    report["air_and_dust"] = measure(
        lambda: env.step(velocity, 0.02, p[0], q[0], 1), 50
    )
    rgb = np.zeros((192, 256, 3), np.uint8)
    depth = np.full((192, 256), 4, np.float32)
    report["aerosol"] = measure(lambda: env.camera(rgb, depth, p[0], q[0]), 10)
    report["optics"] = measure(
        lambda: env.optics.apply(rgb), 20, sync=torch.mps.synchronize
    )
    # The training benchmark consumes recorded RGB-D, proprioception and teacher actions.
    data_path = Path("artifacts/environment-engine-20260905/utility_plant-training.npz")
    data = np.load(data_path)
    for batch in (16, 64, 128):
        ix = np.arange(batch * 8) % len(data["rgb"])
        images = image_tensor(data["rgb"][ix], data["depth"][ix], "mps").reshape(
            batch, 8, 4, 48, 64
        )
        proprio = torch.tensor(data["proprio"][ix].reshape(batch, 8, 11), device="mps")
        targets = torch.tensor(data["teacher"][ix].reshape(batch, 8, 4), device="mps")
        model = VisualController().to("mps")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

        def update():
            prediction, _ = model(images, proprio)
            loss = (prediction - targets).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        report["training"].append(
            {
                "batch": batch,
                "sequence": 8,
                **measure(update, 30, batch * 8, torch.mps.synchronize),
                "mps_allocated_bytes": torch.mps.current_allocated_memory(),
            }
        )
    start = perf_counter()
    result, *_ = run_mission(scene, industrial=True, ticks=30, seed=400)
    report["end_to_end"] = {
        "wall_s": perf_counter() - start,
        "sensor_ticks": result["ticks"],
        "sensor_ticks_per_s": result["ticks"] / (perf_counter() - start),
        "scope": "classical controller, RGB-D, dust, native dynamics and in-memory recording; no disk export",
    }
    report["process_peak_rss_bytes"] = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss
    report["scope"] = (
        "Single process; native batch scaling, not independent parallel rollout workers. Training throughput is optimizer samples/s, not unique data generation."
    )
    path = Path("artifacts/generalization-20260905")
    path.mkdir(exist_ok=True)
    (path / "current-throughput.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
