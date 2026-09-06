"""Image/depth visual-servo teacher used only for imitation labels."""

import numpy as np
from flightrl import _binding
from flightrl.fleet.vehicles import VEHICLES
from .sensors import rotations


def decisions(packet):
    rgb = packet.rgb.astype(float)
    n = len(rgb)
    velocity = np.zeros((n, 3), np.float32)
    yaw = np.zeros(n, np.float32)
    found = np.zeros(n, np.float32)
    for i in range(n):
        role = int(packet.role[i].argmax())
        channel = [0, 2, 1][role]
        image = rgb[i]
        other = np.delete(image, channel, axis=-1).max(-1)
        mask = (image[..., channel] > 90) & (image[..., channel] > other * 1.6 + 25)
        rows, cols = np.where(mask)
        if len(rows) < 3:
            yaw[i] = 0.25
            continue
        distance = float(np.median(packet.depth[i][mask]))
        lateral = -(float(cols.mean()) - 31.5) / 39.1 * distance
        vertical = -(float(rows.mean()) - 23.5) / 39.1 * distance
        velocity[i] = [
            np.clip((distance - 1.05) * 0.5, -0.25, 0.5),
            np.clip(lateral * 0.7, -0.45, 0.45),
            np.clip(vertical * 0.7, -0.5, 0.5),
        ]
        yaw[i] = np.clip(lateral * 0.2, -0.3, 0.3)
        found[i] = float(
            distance < 1.35 and abs(lateral) < 0.25 and abs(vertical) < 0.25
        )
        center = float(np.quantile(packet.depth[i, 18:30, 26:38], 0.1))
        if center < 0.55:
            velocity[i, 0] = min(velocity[i, 0], 0)
        if role == 2 and not np.all(
            packet.messages[i, :2] * packet.messages[i, 2:4] > 0.5
        ):
            velocity[i] = 0
            yaw[i] = 0
            found[i] = 0
    return velocity, yaw, found


def labels(packet, q):
    desired, yaw, found = decisions(packet)
    # The only native controller inputs are sensor-equivalent velocity/attitude.
    world_v = np.einsum("nij,nj->ni", rotations(q), packet.proprio[:, :3]).astype(
        np.float32
    )
    commands = np.ascontiguousarray(np.c_[desired / 0.7, yaw / 2.5], np.float32)
    commands[:, 2] = desired[:, 2] / 0.4
    output = np.empty((len(q), 4), np.float32)
    physics = np.repeat(VEHICLES["fpv"].physics()[None], len(q), axis=0)
    _binding.sixdof_setpoint_actions(
        world_v, q, commands, physics, output, 0.7, 0.4, 2.5, 6.0, 3.0
    )
    return output, found
