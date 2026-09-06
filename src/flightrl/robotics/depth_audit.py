"""Independent physical ray checks of raw rendered range, never controller input."""

import mujoco as mj
import numpy as np


def depth_audit(world, frames, captured):
    data = mj.MjData(world.model)
    data.qpos[:] = captured["qpos"]
    data.mocap_pos[:] = np.array(captured["mocap_pos"]).reshape(data.mocap_pos.shape)
    data.mocap_quat[:] = np.array(captured["mocap_quat"]).reshape(data.mocap_quat.shape)
    mj.mj_forward(world.model, data)
    errors = []
    discrepancies = []
    for i, (name, body) in enumerate(zip(world.camera_names, world.camera_bodies)):
        depth = frames[i][1][1]
        height, width = depth.shape
        f = height / (2 * np.tan(np.deg2rad(63) / 2))
        site = mj.mj_name2id(world.model, mj.mjtObj.mjOBJ_SITE, name)
        rotation = data.site_xmat[site].reshape(3, 3)
        for y in range(8, height, 16):
            for x in range(8, width, 16):
                ray = rotation @ np.array(
                    [1, -(x + 0.5 - width / 2) / f, -(y + 0.5 - height / 2) / f]
                )
                ray /= np.linalg.norm(ray)
                hit = np.array([-1], np.int32)
                distance = mj.mj_ray(
                    world.model,
                    data,
                    data.site_xpos[site],
                    ray,
                    None,
                    1,
                    body,
                    hit,
                )
                expected = 8 if distance < 0 else min(8, distance)
                errors.append(abs(float(depth[y, x]) - expected))
                if errors[-1] >= 0.006:
                    discrepancies.append(
                        dict(
                            robot=i,
                            pixel=[x, y],
                            rendered_m=float(depth[y, x]),
                            physical_m=expected,
                            geom=mj.mj_id2name(
                                world.model, mj.mjtObj.mjOBJ_GEOM, int(hit[0])
                            )
                            if hit[0] >= 0
                            else None,
                            neighbors_m=depth[
                                max(0, y - 1) : y + 2, max(0, x - 1) : x + 2
                            ].tolist(),
                        )
                    )
    return dict(
        sequence=captured["sequence"],
        rays=len(errors),
        max_error_m=max(errors),
        p95_error_m=float(np.quantile(errors, 0.95)),
        fraction_within_6mm=float(np.mean(np.array(errors) < 0.006)),
        discrepancies=discrepancies,
    )
