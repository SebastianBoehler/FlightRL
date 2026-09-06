"""A prescribed 3-D over/under course for testing learned low-level control."""

from types import SimpleNamespace
import numpy as np
from flightrl.fleet.cooperative.dynamics import Flight


def run(policy, seed):
    rng = np.random.default_rng(seed)
    high = float(rng.uniform(2.1, 2.5))
    low = float(rng.uniform(0.65, 0.9))
    room = np.array([-4, 8, -4, 4, 0, 5, 10], np.float32)
    boxes = np.array([[0, 0.4, -4, 4, 0, 1.4], [3, 3.4, -4, 4, 1.6, 5]], np.float32)
    scene = SimpleNamespace(
        scenario=SimpleNamespace(
            arrays={"terrain_bounds": room, "terrain_obstacles": boxes}
        )
    )
    homes = np.array([[-2, y, 0.8] for y in (-2, 0, 2)], np.float32)
    flight = Flight(scene, homes)
    routes = []
    for x, y, z in homes:
        outward = [
            [-0.8, y, high],
            [1.3, y, high],
            [2, y, low],
            [4.2, y, low],
            [5.3, y + 0.35, 3.1],
        ]
        routes.append(outward + outward[-2::-1] + [[x, y, z]])
    indices = np.zeros(3, int)
    records = []
    headings = np.zeros(3, np.float32)
    hit = False
    for tick in range(1600):
        goals = np.array(
            [routes[i][min(indices[i], len(routes[i]) - 1)] for i in range(3)],
            np.float32,
        )
        delta = goals - flight.p
        for i in range(3):
            if np.linalg.norm(delta[i, :2]) > 0.3:
                headings[i] = np.arctan2(delta[i, 1], delta[i, 0])
        completed = indices >= len(routes[0])
        records.append(
            {
                "time_s": round(tick * 0.1, 2),
                "positions": flight.p.tolist(),
                "quaternions": flight.q.tolist(),
                "goals": goals.tolist(),
                "completed": completed.tolist(),
            }
        )
        if completed.all():
            break
        flight.actions[:] = policy(delta, flight.v, flight.q, headings)
        hit = flight.integrate()
        if hit:
            break
        for i in range(3):
            if (
                indices[i] < len(routes[i])
                and np.linalg.norm(flight.p[i] - goals[i]) < 0.18
                and np.linalg.norm(flight.v[i]) < 0.2
            ):
                indices[i] += 1
    z = np.array([r["positions"] for r in records])[:, :, 2]
    status = "collision" if hit else "complete" if completed.all() else "timeout"
    return {
        "records": records,
        "scene": {"room": room.tolist(), "boxes": boxes.tolist()},
        "result": {
            "status": status,
            "controller": "Learned direct thrust and body rates",
            "mission_time_s": round(flight.physics_steps * 0.02, 2),
            "last_capture_time_s": records[-1]["time_s"],
            "terminal_positions": flight.p.tolist(),
            "min_peer_clearance_m": flight.min_gap,
            "altitude_range_m": np.ptp(z, axis=0).tolist(),
            "waypoints_reached": indices.tolist(),
        },
        "provenance": {
            "family": "flight_course",
            "seed": seed,
            "mission": "direct_flight",
            "roles": [
                "Over / under course A",
                "Over / under course B",
                "Over / under course C",
            ],
            "scope": "Learned collective thrust and roll/pitch/yaw rates in native six-DOF physics. Prescribed 3-D waypoints; proprioceptive imitation, not camera navigation or learned obstacle planning.",
            "vehicle": "FPV reference · learned low-level control",
            "dimensions": [0.3, 0.3, 0.1],
            "camera": "Live rendering at recorded pose",
            "communication": "Independent controllers · no team messages",
        },
    }
