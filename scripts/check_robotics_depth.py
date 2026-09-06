"""Cross-check collected WebGPU ray depths against independent MuJoCo rays."""

import argparse
import json
from pathlib import Path
import mujoco as mj
import numpy as np
from flightrl.robotics.industrial import equipment
from flightrl.robotics.world import RobotWorld


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = np.load(args.data / "train.npz")
    data = {key: archive[key] for key in ("depth", "seed", "sequence")}
    world = RobotWorld(equipment(11)[0])
    target = np.array(equipment(11)[1][0]["position"])
    rng = np.random.default_rng(11 * 1009)
    selected = [0, 50, 100, 150, 250, 350, 499]
    errors = []
    f = 96 / (2 * np.tan(np.deg2rad(63) / 2))
    site = mj.mj_name2id(world.model, mj.mjtObj.mjOBJ_SITE, "drone_camera")
    for i in range(500):
        distance = rng.uniform(0.85, 1.4) if i % 3 == 0 else rng.uniform(1.1, 5.3)
        world.data.qpos[:3] = target + [
            -distance,
            rng.uniform(-0.35, 0.35),
            rng.uniform(-0.25, 0.25),
        ]
        angles = np.array(
            [rng.uniform(-0.07, 0.07), rng.uniform(-0.1, 0.1), rng.uniform(-0.22, 0.22)]
        )
        mj.mju_euler2Quat(world.data.qpos[3:7], angles, "xyz")
        world.data.qvel[:6] = rng.uniform(-0.2, 0.2, 6)
        if i not in selected:
            continue
        assert data["seed"][i] == 11 and data["sequence"][i] == i
        mj.mj_forward(world.model, world.data)
        rotation = world.data.xmat[world.drone].reshape(3, 3)
        for y in range(8, 96, 16):
            for x in range(8, 128, 16):
                ray = rotation @ np.array([1, -(x + 0.5 - 64) / f, -(y + 0.5 - 48) / f])
                ray /= np.linalg.norm(ray)
                hit = np.array([-1], np.int32)
                distance = mj.mj_ray(
                    world.model,
                    world.data,
                    world.data.site_xpos[site],
                    ray,
                    None,
                    1,
                    world.drone,
                    hit,
                )
                expected = 8 if distance < 0 else min(8, distance)
                errors.append(abs(float(data["depth"][i, y, x]) - expected))
    report = dict(
        frames=len(selected),
        rays=len(errors),
        max_error_m=max(errors),
        p95_error_m=float(np.quantile(errors, 0.95)),
        tolerance_m=0.006,
        passed=max(errors) < 0.006,
        note="Actual saved float16 WebGPU depth compared with MuJoCo geometry; raster edge pixels are not excluded",
    )
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
