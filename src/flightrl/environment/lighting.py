"""Scene lighting for aerosol scattering; cached on the static concentration grid."""

import numpy as np


def unoccluded(origins, targets, boxes):
    delta = targets - origins
    visible = np.ones(len(origins), bool)
    for b in boxes:
        near = np.zeros(len(origins))
        far = np.ones(len(origins))
        hit = np.ones(len(origins), bool)
        for k in range(3):
            moving = abs(delta[:, k]) > 1e-10
            hit &= moving | (
                (origins[:, k] >= b[2 * k]) & (origins[:, k] <= b[2 * k + 1])
            )
            a = np.full(len(origins), -np.inf)
            z = np.full(len(origins), np.inf)
            np.divide(b[2 * k] - origins[:, k], delta[:, k], out=a, where=moving)
            np.divide(b[2 * k + 1] - origins[:, k], delta[:, k], out=z, where=moving)
            near = np.maximum(near, np.minimum(a, z))
            far = np.minimum(far, np.maximum(a, z))
        visible &= ~(hit & (near <= far) & (far > 1e-4) & (near < 0.9999))
    return visible


def volume_lighting(profile, points, boxes, room):
    illumination = np.full(points.shape, profile.ambient)
    for light in profile.lights:
        targets = np.broadcast_to(light[:3], points.shape)
        distance2 = np.sum((targets - points) ** 2, axis=1)
        power = light[6] / (1 + 0.35 * distance2) * unoccluded(points, targets, boxes)
        illumination += power[:, None] * np.array(light[3:6]) * 0.3
    direction = np.array(profile.sun_direction) / np.linalg.norm(profile.sun_direction)
    distance = np.full(len(points), np.inf)
    for k in range(3):
        if abs(direction[k]) > 1e-8:
            boundary = room[2 * k + int(direction[k] > 0)]
            distance = np.minimum(distance, (boundary - points[:, k]) / direction[k])
    exit_points = points + distance[:, None] * direction
    through_window = np.full(len(points), profile.surface_style == "forest", bool)
    for window in profile.windows:
        through_window |= np.all(
            (exit_points >= np.array(window)[::2] - 0.02)
            & (exit_points <= np.array(window)[1::2] + 0.02),
            axis=1,
        )
    sunlight = (
        through_window * unoccluded(points, exit_points, boxes) * profile.sun_strength
    )
    illumination += sunlight[:, None] * 0.3
    return np.clip(illumination * np.array([158, 153, 139]), 0, 255)
