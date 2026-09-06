"""Three camera-driven aircraft; truth is confined to simulation and scoring."""

from types import SimpleNamespace
import numpy as np
from flightrl import _binding
from flightrl.fleet.cooperative.dynamics import Flight
from .data import COLORS, APPEARANCE
from .sensors import CameraPacket, proprioception
from .teacher import labels


def run(
    seed,
    policy=None,
    ablation=None,
    save_images=False,
    ticks=500,
    samples_out=None,
    scene_setup=None,
):
    rng = np.random.default_rng(seed)
    room = np.array([-3, 7, -4, 4, 0, 4, 8], np.float32)
    # The visual target heights demand ascent and descent. Obstacles share the
    # exact native geometry rendered into the actor's RGB-D observations.
    boxes = np.array(
        [[1.6, 1.9, -3, -1.1, 0, 0.8], [1.6, 1.9, -0.9, 0.9, 2.2, 4]], np.float32
    )
    panels = np.array(
        [
            [
                5,
                y,
                float(z + rng.uniform(-0.12, 0.12)),
                0,
                -1,
                0,
                0,
                0,
                1,
                0.3,
                0.3,
                *color,
            ]
            for y, z, color in zip([-2, 0, 2], [2.1, 0.9, 1.8], COLORS)
        ],
        np.float32,
    )
    appearance = APPEARANCE
    if scene_setup is not None:
        room, boxes, panels, appearance = scene_setup(room, boxes, panels)
    home = np.array([[-0.5, y, 1.3] for y in [-2, 0, 2]], np.float32)
    home[:, :2] += rng.uniform(-0.12, 0.12, (3, 2)).astype(np.float32)
    flight = Flight(
        SimpleNamespace(
            scenario=SimpleNamespace(
                arrays={"terrain_bounds": room, "terrain_obstacles": boxes}
            )
        ),
        home,
    )
    rgb = np.empty((3, 48, 64, 3), np.uint8)
    depth = np.empty((3, 48, 64), np.float32)
    reports = np.zeros(3, bool)
    delivered = np.zeros(2, bool)
    reported_at = np.full(3, np.inf)
    dwell = np.zeros(3, int)
    events = []
    records = []
    images = []
    hit = False
    false_report = False
    for tick in range(ticks):
        now = tick * 0.1
        for i in range(3):
            peer_boxes = np.array(
                [
                    np.column_stack((flight.p[j] - 0.14, flight.p[j] + 0.14)).ravel()
                    for j in range(3)
                    if j != i
                ],
                np.float32,
            )
            _binding.inspection_render(
                flight.p[i : i + 1],
                flight.q[i : i + 1],
                room,
                np.concatenate([boxes, peer_boxes]),
                panels,
                rgb[i : i + 1],
                np.empty((1, 3, 2), np.int32),
                depth[i : i + 1],
                1,
                *appearance,
            )
        delivered = (
            (now >= reported_at[:2] + 0.2)
            if ablation != "no_messages"
            else np.zeros(2, bool)
        )
        messages = np.tile(np.r_[delivered, [1, 1], [0.2, 0.2]], (3, 1)).astype(
            np.float32
        )
        if ablation == "no_messages":
            messages[:] = 0
        packet = CameraPacket(
            rgb.copy(),
            depth.copy(),
            proprioception(flight.v, flight.q, flight.rates),
            np.eye(3, dtype=np.float32),
            messages,
            tick,
            now,
        )
        if ablation == "no_images":
            packet = CameraPacket(
                np.zeros_like(rgb),
                np.zeros_like(depth),
                packet.proprio,
                packet.role,
                packet.messages,
                tick,
                now,
            )
        if samples_out is not None:
            expected, detection = labels(packet, flight.q)
            samples_out.append((packet, expected.copy(), detection.copy()))
        if policy is None:
            action, confidence = labels(packet, flight.q)
        else:
            action, confidence = policy(packet)
        dwell = np.where(confidence > 0.8, dwell + 1, 0)
        for i in range(3):
            if not reports[i] and dwell[i] >= 3:
                reports[i] = True
                reported_at[i] = now
                correct = np.linalg.norm(flight.p[i] - panels[i, :3]) < 1.6 and (
                    i < 2 or delivered.all()
                )
                false_report |= not correct
                events.append(
                    {
                        "time_s": now,
                        "type": "detected" if i < 2 else "confirmed",
                        "drone": i,
                        "text": f"Drone {i + 1} visual report"
                        + (" (incorrect)" if not correct else ""),
                    }
                )
        records.append(
            {
                "time_s": round(now, 2),
                "positions": flight.p.tolist(),
                "quaternions": flight.q.tolist(),
                "goals": panels[:, :3].tolist(),
                "completed": reports.tolist(),
            }
        )
        if save_images:
            images.append(rgb.copy())
        if reports.all() or false_report:
            break
        flight.actions[:] = action
        hit = flight.integrate()
        if hit:
            break
    status = (
        "collision"
        if hit
        else "incorrect_report"
        if false_report
        else "complete"
        if reports.all()
        else "timeout"
    )
    result = {
        "status": status,
        "controller": "RGB-D role-conditioned CTBR actor"
        if policy
        else "sensor-only visual teacher",
        "mission_time_s": round(flight.physics_steps * 0.02, 2),
        "last_capture_time_s": records[-1]["time_s"],
        "terminal_positions": flight.p.tolist(),
        "reports": reports.tolist(),
        "min_peer_clearance_m": flight.min_gap,
        "altitude_range_m": np.ptp(
            np.array([r["positions"] for r in records])[:, :, 2], axis=0
        ).tolist(),
    }
    replay = {
        "records": records,
        "events": events,
        "scene": {
            "room": room.tolist(),
            "boxes": boxes.tolist(),
            "panels": panels.tolist(),
        },
        "result": result,
        "provenance": {
            "family": "camera_control",
            "mission": "camera_control",
            "seed": seed,
            "roles": [
                "Red beacon scout",
                "Blue beacon scout",
                "Green beacon confirmer",
            ],
            "scope": "Raw RGB-D + ideal body odometry/IMU + role + delayed visual reports. No actor position, target coordinates, waypoints or obstacle map. Synthetic color beacons, not people.",
            "vehicle": "FPV reference · sensor-only learned direct CTBR",
            "dimensions": [0.3, 0.3, 0.1],
            "camera": "Recorded actor RGB · 64 × 48",
            "communication": "Scout visual reports delayed 200 ms; one shared policy",
        },
    }
    return replay, images
