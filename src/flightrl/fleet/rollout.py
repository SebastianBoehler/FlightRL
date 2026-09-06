"""Three simultaneous native drones with local RGB-D and delayed peer telemetry."""

from time import perf_counter
import numpy as np
from flightrl import _binding
from flightrl.sixdof.native import native_step
from .vehicles import VEHICLES
from .communication import PeerLink, peer_features


def free_points(room, boxes, radius, rng, count):
    points = []
    for _ in range(10000):
        p = rng.uniform(room[::2][:3] + radius + 0.15, room[1::2][:3] - radius - 0.15)
        if any(
            np.all(p >= b[::2] - radius) and np.all(p <= b[1::2] + radius)
            for b in boxes
        ):
            continue
        if any(np.linalg.norm(p - q) < 2 * radius + 0.5 for q in points):
            continue
        points.append(p)
        if len(points) == count:
            return np.array(points, np.float32)
    raise ValueError("vehicle does not fit enough separated spawn/goal locations")


def run_fleet(
    scene,
    vehicle_key,
    mission,
    seed,
    ticks=200,
    policy=None,
    collect=False,
    dropout=0.0,
):
    if mission not in ("inspect", "delivery", "return"):
        raise ValueError("unknown fleet mission")
    vehicle = VEHICLES[vehicle_key]
    room = scene.scenario.arrays["terrain_bounds"]
    boxes = scene.scenario.arrays["terrain_obstacles"]
    radius = vehicle.radius
    if np.any(room[1:6:2] - room[:6:2] <= 2 * radius + 0.3):
        return {"status": "incompatible_envelope", "vehicle": vehicle_key}, [], []
    rng = np.random.default_rng(seed)
    try:
        sites = free_points(room, boxes, radius, rng, 6)
    except ValueError:
        return {"status": "incompatible_envelope", "vehicle": vehicle_key}, [], []
    n = 3
    p = sites[:n].copy()
    home = p.copy()
    goal = sites[n:].copy()
    estimate = p.copy()
    v = np.zeros((n, 3), np.float32)
    q = np.zeros((n, 4), np.float32)
    q[:, 0] = 1
    rates = v.copy()
    thrust = np.ones(n, np.float32)
    ranges = np.empty((n, 6), np.float32)
    physics = np.repeat(vehicle.physics()[None], n, axis=0)
    action = np.zeros((n, 4), np.float32)
    rgb = np.empty((n, 48, 64, 3), np.uint8)
    depth = np.empty((n, 48, 64), np.float32)
    counts = np.empty((1, len(scene.panels), 2), np.int32)
    link = PeerLink(n, drop_probability=dropout, seed=seed)
    completed = [set() for _ in range(n)]
    reached = np.zeros(n, bool)
    targets = np.arange(n)
    data = []
    records = []
    collisions = False
    min_gap = 1e9
    start = perf_counter()
    dwell = np.zeros(n)
    for tick in range(ticks):
        now = tick * 0.1
        measured = v + rng.normal(0, 0.002, (n, 3)).astype(np.float32)
        if tick:
            estimate += measured * 0.1
        if tick % 2 == 0 and not (mission == "return" and now >= 8):
            link.publish(now, estimate, measured, targets, completed)
        peers = link.receive(now)
        if mission == "return" and now >= 8:
            goal = home.copy()
        # Other vehicles are visible to each drone's camera, not only to telemetry.
        for i in range(n):
            peer_boxes = np.array(
                [
                    np.column_stack((p[j] - radius, p[j] + radius)).ravel()
                    for j in range(n)
                    if j != i
                ],
                np.float32,
            )
            obstacles = np.ascontiguousarray(
                np.concatenate((boxes, peer_boxes)), np.float32
            )
            _binding.inspection_render(
                p[i : i + 1],
                q[i : i + 1],
                room,
                obstacles,
                scene.panels,
                rgb[i : i + 1],
                counts,
                depth[i : i + 1],
                1,
                *scene.environment.render_buffers(),
            )
        commands = []
        for i in range(n):
            delta = goal[i] - estimate[i]
            yaw = 2 * np.arctan2(q[i, 3], q[i, 0])
            c, s = np.cos(yaw), np.sin(yaw)
            body = np.array(
                [c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1], delta[2]]
            )
            bearing = np.arctan2(body[1], body[0])
            distance = float(np.linalg.norm(delta))
            dwell[i] = dwell[i] + 0.1 if distance < 0.4 else 0
            required = 1.5 if mission == "inspect" else 0.3
            if dwell[i] >= required and (mission != "return" or now >= 8):
                reached[i] = True
                completed[i].add(int(targets[i]))
            state = np.r_[
                measured[i],
                q[i],
                np.clip(body, -8, 8) / 8,
                radius,
                vehicle.mass_kg / 32,
                vehicle.motor_tau_s,
                vehicle.rate_tau_s,
                peer_features(peers[i], estimate[i], measured[i], now),
                [mission == m for m in ("inspect", "delivery", "return")],
            ].astype(np.float32)
            clearance = float(np.quantile(depth[i, 18:30, 26:38], 0.1))
            teacher = np.array(
                [
                    min(0.65, distance) * max(0, np.cos(bearing)),
                    0,
                    np.clip(body[2], -0.3, 0.3),
                    np.clip(bearing, -0.8, 0.8),
                ],
                np.float32,
            )
            if clearance < radius + 0.8:
                teacher[0] *= np.clip((clearance - radius - 0.2) / 0.6, 0, 1)
                teacher[3] = (
                    0.6 if depth[i, :, 0:20].mean() > depth[i, :, 44:].mean() else -0.6
                )
            for msg in peers[i]:
                separation = estimate[i] - np.array(msg.position)
                if np.linalg.norm(separation) < 2 * radius + 0.7:
                    teacher[0] = 0
                    teacher[2] = 0.2 if i > msg.sender else -0.2
            if reached[i]:
                teacher[:] = 0
            if collect:
                data.append(
                    (rgb[i].copy(), depth[i].copy(), state.copy(), teacher.copy())
                )
            command = teacher if policy is None else policy(rgb[i], depth[i], state)
            # Shared camera brake; learned actions otherwise drive all four axes.
            command = np.clip(
                command, [-0.4, -0.4, -0.3, -0.8], [0.65, 0.4, 0.3, 0.8]
            ).astype(np.float32)
            command[0] *= np.clip((clearance - radius - 0.15) / 0.4, 0, 1)
            if reached[i]:
                command[:] = 0
            commands.append(command)
        records.append(
            {
                "time_s": now,
                "positions": p.tolist(),
                "quaternions": q.tolist(),
                "goals": goal.tolist(),
                "completed": reached.tolist(),
            }
        )
        _binding.sixdof_setpoint_actions(
            v,
            q,
            np.array(commands, np.float32),
            physics,
            action,
            0.65,
            0.4,
            2.5,
            6.0,
            3.0,
        )
        for _ in range(5):
            before = p.copy()
            native_step(p, v, q, rates, ranges, action, 0.02, room, thrust, physics)
            hit = np.zeros(n, np.uint8)
            _binding.inspection_collision(before, p, room, boxes, radius, hit)
            collisions |= bool(hit.any())
            for i in range(n):
                for j in range(i):
                    a = before[i] - before[j]
                    d = (p[i] - p[j]) - a
                    t = np.clip(-np.dot(a, d) / max(np.dot(d, d), 1e-12), 0, 1)
                    gap = float(np.linalg.norm(a + t * d) - 2 * radius)
                    min_gap = min(min_gap, gap)
                    collisions |= gap <= 0
            if collisions:
                break
        if collisions or reached.all():
            break
    wall = perf_counter() - start
    return (
        {
            "status": "collision"
            if collisions
            else "complete"
            if reached.all()
            else "budget_exhausted",
            "coverage": float(reached.mean()),
            "collision": collisions,
            "vehicle": vehicle_key,
            "mission": mission,
            "ticks": len(records),
            "wall_s": wall,
            "agent_camera_steps_per_s": n * len(records) / wall,
            "minimum_peer_surface_gap_m": min_gap,
            "messages_delivered": link.delivered,
            "controller": "visual_fleet_student" if policy else "local_rgbd_teacher",
        },
        records,
        data,
    )
