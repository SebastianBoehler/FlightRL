"""Simulation/evaluation harness. Truth never enters VisualOdometry or SurfaceMap."""

import time
import numpy as np
from flightrl import _binding
from flightrl.environment.profile import EnvironmentProfile
from flightrl.fleet.camera_policy.episode import run
from flightrl.fleet.camera_policy.sensors import rotations
from .geometry import intrinsics, transform, dense_points
from .odometry import VisualOdometry
from .fusion import SurfaceMap
from .metrics import score
from .registration import register

OPTICAL_TO_BODY = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])


def setup(seed):
    """Unseen equipment placement and appearance, with original mission beacons."""
    rng = np.random.default_rng(seed)
    appearance = EnvironmentProfile(
        "mapping-test",
        ambient=float(rng.uniform(0.32, 0.55)),
        equipment_rgb=(65, 72, 81),
        floor_rgb=(95, 85, 75),
    ).render_buffers()

    def build(room, boxes, panels):
        extra = np.array(
            [
                [x, x + 0.5, 3.0, 3.6, 0, h]
                for x, h in zip(rng.uniform(0, 5, 4), rng.uniform(0.5, 2.8, 4))
            ],
            np.float32,
        )
        return room, np.concatenate([boxes, extra]), panels, appearance

    return build, appearance


def camera_pose(p, q):
    r = rotations(np.array([q], np.float32))[0]
    result = np.eye(4)
    result[:3, :3] = r @ OPTICAL_TO_BODY
    result[:3, 3] = p + r @ np.array([0.035, 0, 0.012])
    return result


def experiment(seed, policy, ticks=500, reference_out=None):
    build, appearance = setup(seed)
    start = time.perf_counter()
    replay, _ = run(seed, policy=policy, scene_setup=build, ticks=ticks)
    flight_seconds = time.perf_counter() - start
    scene = replay["scene"]
    k = intrinsics()
    room, boxes, panels = [
        np.array(scene[key], np.float32) for key in ("room", "boxes", "panels")
    ]
    rgb = np.empty((1, 192, 256, 3), np.uint8)
    depth = np.empty((1, 192, 256), np.float32)
    counts = np.empty((1, len(panels), 2), np.int32)
    records = replay["records"]
    first = records[0]
    origins = [
        camera_pose(p, q) for p, q in zip(first["positions"], first["quaternions"])
    ]
    inverse = [np.linalg.inv(p) for p in origins]
    backends = {
        (i, mode): VisualOdometry(k, mode) for i in range(3) for mode in ("rgb", "rgbd")
    }
    maps = {key: SurfaceMap() for key in backends}
    poses = {key: [] for key in backends}
    truth = [[] for _ in range(3)]
    references = [[] for _ in range(3)]
    initial = []
    frames = []
    start = time.perf_counter()
    for frame, record in enumerate(records):
        pictures = []
        for i in range(3):
            p = np.array(record["positions"][i : i + 1], np.float32)
            q = np.array(record["quaternions"][i : i + 1], np.float32)
            peers = np.array(
                [
                    np.column_stack((np.array(v) - 0.14, np.array(v) + 0.14)).ravel()
                    for j, v in enumerate(record["positions"])
                    if j != i
                ],
                np.float32,
            )
            _binding.inspection_render(
                p,
                q,
                room,
                np.concatenate([boxes, peers]),
                panels,
                rgb,
                counts,
                depth,
                1,
                *appearance,
            )
            pictures.append(rgb[0].copy())
            if frame == 0:
                initial.append((rgb[0].copy(), depth[0].copy()))
            gt = inverse[i] @ camera_pose(p[0], q[0])
            truth[i].append(gt)
            points, _ = dense_points(depth[0], k)
            references[i].append(transform(points, gt))
            for mode in ("rgb", "rgbd"):
                key = (i, mode)
                backend = backends[key]
                pose = backend.step(rgb[0], depth[0] if mode == "rgbd" else None)
                poses[key].append(pose)
                if pose is None:
                    continue
                if mode == "rgbd":
                    maps[key].integrate(rgb[0], depth[0], k, pose, frame + 1)
                elif backend.landmarks is not None:
                    xy = np.rint(backend.pixels).astype(int)
                    xy = np.clip(xy, [0, 0], [255, 191])
                    maps[key].add(
                        backend.landmarks, rgb[0, xy[:, 1], xy[:, 0]], frame + 1
                    )
        frames.append(np.concatenate(pictures, axis=1))
    elapsed = time.perf_counter() - start
    if reference_out is not None:
        reference_out.extend(np.concatenate(samples) for samples in references)
    results = []
    payload = []
    for (i, mode), surface in maps.items():
        cloud = surface.export()
        metrics = score(
            poses[i, mode],
            np.array(truth[i]),
            cloud,
            np.concatenate(references[i]),
            mode,
        )
        results.append(dict(drone=i + 1, mode=mode, points=len(cloud), **metrics))
        payload.append(
            dict(
                drone=i + 1,
                mode=mode,
                points=cloud,
                poses=[
                    None if p is None else p[:3, 3].tolist() for p in poses[i, mode]
                ],
                truth=[p[:3, 3].tolist() for p in truth[i]],
                metrics=metrics,
            )
        )
    registrations = []
    for i in (1, 2):
        relative, n = register(initial[0][0], initial[0][1], initial[i][0], k)
        entry = dict(drone=i + 1, accepted=relative is not None, inliers=n)
        if relative is not None:
            gt = inverse[0] @ origins[i]
            entry["translation_error_m"] = float(
                np.linalg.norm(relative[:3, 3] - gt[:3, 3])
            )
            entry["rotation_error_deg"] = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            (np.trace(relative[:3, :3].T @ gt[:3, :3]) - 1) / 2, -1, 1
                        )
                    )
                )
            )
        registrations.append(entry)
    result = dict(
        seed=seed,
        mission=replay["result"],
        mapping=results,
        registration=registrations,
        flight_wall_s=flight_seconds,
        reconstruction_wall_s=elapsed,
        camera_frames=len(records) * 3,
        camera_frames_per_s=len(records) * 3 / elapsed,
        coverage_definition="Fraction of sampled surfaces visible along the flown trajectory within 0.15 m of reconstruction; NOT whole-site coverage",
        mapping_completion="Not assessed: active exploration and whole-site coverage are not implemented",
    )
    return (
        result,
        dict(seed=seed, dt=0.1, frames=len(records), maps=payload, result=result),
        frames,
    )
