"""Behavior gates for task ownership, failure takeover and conservative routes."""

import numpy as np
from flightrl.fleet.cooperative.mission import run
from flightrl.fleet.cooperative.routing import Routes
from flightrl.inspection.environments import environment_scene


def test_failure_takeover_finishes_each_target_once():
    replay = run(20)
    result = replay["result"]
    assert result["status"] == "complete"
    assert result["minimum_peer_surface_gap_m"] > 0
    assert sorted(t for tasks in result["completed_by"] for t in tasks) == list(
        range(9)
    )
    unavailable = next(e for e in replay["events"] if e["type"] == "unavailable")
    release = next(e for e in replay["events"] if e["type"] == "release")
    takeover = next(e for e in replay["events"] if e["type"] == "takeover")
    assert release["time_s"] >= unavailable["time_s"] + 0.2 - 1e-6
    assert takeover["task"] == release["task"] and takeover["drone"] != 1
    assert all(
        e["time_s"] < 8
        for e in replay["events"]
        if e["type"] == "inspection" and e["drone"] == 1
    )
    for frame in replay["records"]:
        assigned = [j for j in frame["assignments"] if j >= 0]
        assert len(set(assigned)) == len(assigned)


def test_fixed_assignments_leave_failed_drone_work_unfinished():
    result = run(20, mode="fixed", ticks=900, record=False)["result"]
    assert result["status"] == "budget_exhausted"
    assert result["completed_targets"] < 9
    assert result["takeovers"] == 0


def test_compressed_routes_preserve_obstacle_clearance():
    r = Routes(environment_scene("forest", 20))
    sites = r.sites(20)
    for point in sites[1:]:
        path = r.path(sites[0], point)
        assert np.allclose(path[-1], point)
        for a, b in zip(path, path[1:]):
            assert np.count_nonzero(np.abs(b - a) > 1e-6) == 1
            for p in np.linspace(a, b, 100):
                assert r.cell(p) in r.free


def test_collision_clock_counts_only_executed_physics_steps():
    from flightrl.fleet.cooperative.dynamics import Flight

    flight = Flight(
        environment_scene("forest", 20), [[0, -2, 0.15], [0, 0, 0.15], [0, 2, 0.15]]
    )
    flight.v[:, 2] = -3
    assert flight.integrate()
    assert flight.physics_steps == 1
