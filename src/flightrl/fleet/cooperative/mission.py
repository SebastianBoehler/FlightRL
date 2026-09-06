"""Learned bids with an explicit coordinator, route planner and failure takeover."""

from time import perf_counter
import numpy as np
from flightrl.inspection.environments import environment_scene
from .routing import Routes
from .dynamics import Flight
from .search import SearchProtocol


def run(
    seed,
    bid=None,
    mode="dynamic",
    failure_s=8.0,
    ticks=2400,
    record=True,
    mission="inspection",
    family="forest",
):
    if mission not in ("inspection", "search_rescue"):
        raise ValueError("Unknown mission")
    if mission == "search_rescue" and failure_s is not None:
        raise ValueError("Search role experiment requires all three drones")
    if mode not in ("dynamic", "fixed"):
        raise ValueError("Unknown coordinator mode")
    scene = environment_scene(family, seed)
    route = Routes(scene)
    sites = route.sites(seed)
    heights = np.array([1.0, 1.8, 2.6])
    homes = np.column_stack((sites[:3], heights))
    tasks = sites[3:]
    search = SearchProtocol(len(tasks)) if mission == "search_rescue" else None
    flight = Flight(scene, homes)
    owners = np.full(9, -1)
    done = np.zeros(9, bool)
    assign = np.full(3, -1)
    paths = [[] for _ in range(3)]
    alive = np.ones(3, bool)
    dwell = np.zeros(3)
    goals = homes.copy()
    events = []
    records = []
    returned = np.zeros(3, bool)
    completed_by = [[] for _ in range(3)]
    released = []
    failed = False
    hold = homes.copy()
    start = perf_counter()
    collision = False
    for tick in range(ticks):
        now = round(tick * 0.1, 4)
        if failure_s is not None and now >= failure_s and not failed:
            failed = True
            hold[1] = flight.p[1]
            alive[1] = False
            events.append(
                {
                    "time_s": now,
                    "type": "unavailable",
                    "drone": 1,
                    "text": "Drone 2 unavailable; holding position",
                }
            )
        # Failure reaches the coordinator after the declared 200 ms delay.
        if failed and now >= failure_s + 0.2 and assign[1] >= 0:
            task = int(assign[1])
            released.append(task)
            if mode == "dynamic":
                owners[task] = -1
            assign[1] = -1
            paths[1] = []
            events.append(
                {
                    "time_s": now,
                    "type": "release",
                    "drone": 1,
                    "task": task,
                    "text": f"Target {task + 1} released for takeover"
                    if mode == "dynamic"
                    else f"Target {task + 1} stranded by fixed assignment",
                }
            )
        for i in range(3):
            if not alive[i]:
                goals[i] = hold[i]
                continue
            if assign[i] >= 0:
                j = assign[i]
                settled = (
                    np.linalg.norm(flight.p[i] - goals[i]) < 0.22
                    and np.linalg.norm(flight.v[i]) < 0.18
                )
                dwell[i] = dwell[i] + 0.1 if settled else 0
                if dwell[i] >= 1.0:
                    detection = (
                        search.inspect(i, j, now, flight.p[i], tasks[j], route.boxes)
                        if search
                        else None
                    )
                    if search and detection is None:
                        continue
                    done[j] = not search or i == 2
                    if done[j]:
                        completed_by[i].append(int(j))
                    else:
                        owners[j] = -1
                    assign[i] = -1
                    dwell[i] = 0
                    events.append(
                        {
                            "time_s": now,
                            "type": detection["type"] if detection else "inspection",
                            "drone": i,
                            "task": int(j),
                            "text": detection["text"]
                            if detection
                            else f"Drone {i + 1} inspected target {j + 1}",
                        }
                    )
            if assign[i] < 0:
                candidates = [
                    j
                    for j in range(9)
                    if not done[j]
                    and owners[j] < 0
                    and (mode != "fixed" or j % 3 == i)
                    and (not search or search.eligible(i, j, now))
                ]
                if candidates:

                    def cost(j):
                        return (
                            bid(route, flight.p[i], tasks[j])
                            if bid
                            else route.length(flight.p[i], tasks[j])
                        )

                    j = min(candidates, key=cost)
                    assign[i] = j
                    owners[j] = i
                    returned[i] = False
                    task_height = heights[i] + (
                        0.2 * np.sin(j * 1.7 + i) if search else 0
                    )
                    goals[i] = [*tasks[j], task_height]
                    points = route.path(flight.p[i], tasks[j])
                    paths[i] = [np.array([*p, task_height]) for p in points]
                    events.append(
                        {
                            "time_s": now,
                            "type": "takeover" if j in released else "assignment",
                            "drone": i,
                            "task": int(j),
                            "text": f"Drone {i + 1} "
                            + ("takes over" if j in released else "assigned")
                            + f" target {j + 1}",
                        }
                    )
                elif not paths[i] and np.linalg.norm(goals[i] - homes[i]) > 0.01:
                    goals[i] = homes[i].copy()
                    paths[i] = [
                        np.array([*p, heights[i]])
                        for p in route.path(flight.p[i], homes[i])
                    ]
            returned[i] = (
                assign[i] < 0
                and np.linalg.norm(flight.p[i] - homes[i]) < 0.25
                and np.linalg.norm(flight.v[i]) < 0.18
            )
        waypoints = []
        for i in range(3):
            if not alive[i]:
                waypoints.append(hold[i])
                continue
            while paths[i] and np.linalg.norm(flight.p[i] - paths[i][0]) < 0.13:
                paths[i].pop(0)
            waypoints.append(paths[i][0] if paths[i] else goals[i])
        if record:
            records.append(
                {
                    "time_s": now,
                    "positions": flight.p.tolist(),
                    "quaternions": flight.q.tolist(),
                    "goals": goals.tolist(),
                    "completed": returned.tolist(),
                    "active": alive.tolist(),
                    "assignments": assign.tolist(),
                    "task_done": done.tolist(),
                    "task_owners": owners.tolist(),
                    "completed_by": [list(x) for x in completed_by],
                    **(
                        {"task_found": search.found.tolist(), "roles": search.roles}
                        if search
                        else {}
                    ),
                }
            )
        if done.all() and np.all(returned[alive]):
            break
        collision = flight.step(waypoints)
        if collision:
            break
    status = (
        "collision"
        if collision
        else "complete"
        if done.all() and np.all(returned[alive])
        else "budget_exhausted"
    )
    return {
        "records": records,
        "events": events,
        "tasks": tasks.tolist(),
        "scene": {"boxes": route.boxes.tolist(), "room": route.room.tolist()},
        "result": {
            "status": status,
            "controller": "learned_route_bids_with_classical_navigation"
            if bid
            else "oracle_routes_with_classical_navigation",
            "coverage": float(done.mean()),
            "completed_targets": int(done.sum()),
            "total_targets": 9,
            "mission_time_s": round(flight.physics_steps * 0.02, 2),
            "last_capture_time_s": now,
            "terminal_positions": flight.p.tolist(),
            "collision": collision,
            "minimum_peer_surface_gap_m": flight.min_gap,
            "wall_s": perf_counter() - start,
            "completed_by": completed_by,
            "takeovers": sum(e["type"] == "takeover" for e in events),
            "mode": mode,
            "failure_s": failure_s,
            "seed": seed,
        },
        "provenance": {
            "family": family,
            "mission": mission,
            "scope": "Two scouts search known sectors; synthetic proximity/line-of-sight beacon detection; a third drone confirms reports after 200 ms. Not person recognition or rescue certification."
            if search
            else "Known-map waypoint inspection",
            **({"roles": search.roles} if search else {}),
            "seed": seed,
            "vehicle": "FPV size reference · mission-directed XYZ flight with varying target heights"
            if search
            else "FPV size reference · altitude lanes 1.0 / 1.8 / 2.6 m",
            "communication": "Central task ledger · scout reports delayed 200 ms · simulator pose telemetry"
            if search
            else "Central task ledger · failure notification delayed 200 ms · simulator pose telemetry",
            "camera": "Visual re-render, not policy observations",
            "dimensions": [0.185, 0.212, 0.064],
        },
    }
