"""Measured native/Metal camera parity and actual MPS learner-update workload."""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from flightrl import _binding
from flightrl.inspection_fixture import three_panel_room
from flightrl.inspection.metal import MetalCamera
from flightrl.inspection.student import VisualController, image_tensor


def timing(fn, device=False):
    values = []
    for _ in range(4):
        fn()
    for _ in range(20):
        if device:
            torch.mps.synchronize()
        start = time.perf_counter()
        fn()
        if device:
            torch.mps.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    return {
        "p50_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene = three_panel_room()
    rng = np.random.default_rng(13)
    results = []
    for n in (1, 32, 64):
        p = np.column_stack(
            (rng.uniform(-2, 0, n), rng.uniform(-2, 2, n), np.full(n, 1.5))
        ).astype(np.float32)
        yaw = rng.uniform(-np.pi, np.pi, n)
        q = np.column_stack(
            (np.cos(yaw / 2), np.zeros(n), np.zeros(n), np.sin(yaw / 2))
        ).astype(np.float32)
        rgb = np.zeros((n, 48, 64, 3), np.uint8)
        depth = np.zeros((n, 48, 64), np.float32)
        counts = np.zeros((n, 3, 2), np.int32)

        def cpu():
            _binding.inspection_render(
                p,
                q,
                scene.scenario.arrays["terrain_bounds"],
                scene.scenario.arrays["terrain_obstacles"],
                scene.panels,
                rgb,
                counts,
                depth,
            )

        camera = MetalCamera(scene, n)
        cpu()
        mr, md = camera.render(p, q)
        torch.mps.synchronize()
        rgb_error = int(np.count_nonzero(mr.cpu().numpy() != rgb))
        depth_error = float(np.max(abs(md.cpu().numpy() - depth)))
        if rgb_error or depth_error > 2e-5:
            raise ValueError(f"Metal parity failed: {rgb_error}, {depth_error}")
        model = VisualController().to("mps")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        proprio = torch.zeros((n, 1, 11), device="mps")
        proprio[:, :, 7] = 0.2
        target = torch.zeros((n, 1, 4), device="mps")
        target[:, :, 0] = 0.36

        def update():
            r, d = camera.render(p, q)
            images = image_tensor(r, d, "mps")[:, None]
            prediction, _ = model(images, proprio)
            loss = ((prediction - target) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        results.append(
            {
                "batch": n,
                "rgb_mismatched_values": rgb_error,
                "max_depth_error_m": depth_error,
                "cpu_render_with_evaluator_counts": timing(cpu),
                "metal_render_pose_upload_sync": timing(
                    lambda: camera.render(p, q), True
                ),
                "metal_render_and_rgbd_readback": timing(
                    lambda: [v.cpu() for v in camera.render(p, q)], True
                ),
                "metal_render_input_conversion_and_learner_update": timing(
                    update, True
                ),
            }
        )
    report = {
        "torch": torch.__version__,
        "device": "Apple MPS",
        "results": results,
        "ownership": "Persistent camera outputs owned by PyTorch MPS; no CPU image copy on learner path",
        "copies": "2 CPU pose uploads per call; RGB/depth stay resident for learner; dtype/channel conversion allocates learner input",
        "synchronization": "Explicit MPS synchronize around each measured iteration",
        "scope": "Microbenchmark, not time-to-policy or proof of end-to-end speedup",
        "mps_allocated_bytes": torch.mps.current_allocated_memory(),
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
