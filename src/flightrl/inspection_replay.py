"""Batched native inspection diagnostic; offline observation/event evaluation."""

import numpy as np

from flightrl import _binding
from flightrl.artifact_identity import sha256_file
from flightrl.inspection_scene import CAMERA, render_scene, swept_collision
from flightrl.inspection_mission import InspectionMemory, QUALITY, evaluate_views
from flightrl.scenario_replay import STATE_FIELDS, _require_array
from flightrl.sixdof.native import native_step


def capture_inspection(scene, actions, initial_position_m, connected, dt_s=0.05):
    _require_array(actions, "actions", np.float32, 3)
    ticks, n, width = actions.shape
    if not (1 <= ticks <= 1000 and 1 <= n <= 64 and width == 4) or np.any(
        abs(actions) > 1
    ):
        raise ValueError("normalized actions require (1..1000,1..64,4)")
    _require_array(initial_position_m, "positions", np.float32, 2)
    _require_array(connected, "connected", np.bool_, 2)
    if initial_position_m.shape != (n, 3) or connected.shape != (ticks + 1, n):
        raise ValueError(
            "initial positions or link observations do not match action batch"
        )
    if type(dt_s) not in (int, float) or not np.isfinite(dt_s) or not 0 < dt_s <= 0.05:
        raise ValueError("dt must be within (0,.05]")
    if (
        np.any(scene.scenario.arrays["sensor_parameters"])
        or scene.scenario.arrays["mission_steps"].size
    ):
        raise ValueError(
            "inspection diagnostic supports ideal sensors and no mission rows"
        )
    p = initial_position_m.copy()
    v = np.zeros((n, 3), np.float32)
    q = np.zeros((n, 4), np.float32)
    q[:, 0] = 1
    rates = v.copy()
    thrust = np.ones(n, np.float32)
    ranges = np.empty((n, 6), np.float32)
    physics = np.repeat(scene.scenario.arrays["vehicle_physics"][None, :], n, axis=0)
    if swept_collision(scene, p, p).any():
        raise ValueError("initial body envelope collides with scene")
    frames = np.empty((ticks + 1, n, 48, 64, 3), np.uint8)
    states = np.empty((ticks + 1, n, 14), np.float32)
    counts = np.empty((ticks + 1, n, len(scene.panels), 2), np.int32)
    collisions = np.zeros((ticks, n), np.uint8)
    attempted = np.empty((ticks, n, 3), np.float32)
    previous = np.empty_like(p)
    terminal = False
    for t in range(ticks + 1):
        states[t, :, :3] = p
        states[t, :, 3:6] = v
        states[t, :, 6:10] = q
        states[t, :, 10:13] = rates
        states[t, :, 13] = thrust
        render_scene(scene, p, q, frames[t], counts[t])
        if terminal or t == ticks:
            break
        previous[:] = p
        native_step(
            p,
            v,
            q,
            rates,
            ranges,
            actions[t],
            dt_s,
            scene.scenario.arrays["terrain_bounds"],
            thrust,
            physics,
        )
        attempted[t] = p
        swept_collision(scene, previous, p, collisions[t])
        if collisions[t].any():
            # Terminal contact: retain last valid full state; never bounce or tunnel.
            p[:] = states[t, :, :3]
            v[:] = states[t, :, 3:6]
            q[:] = states[t, :, 6:10]
            rates[:] = states[t, :, 10:13]
            thrust[:] = states[t, :, 13]
            terminal = True
    used = t
    arrays = {
        "time_s": np.arange(used + 1, dtype=np.float64) * dt_s,
        "states_truth": states[: used + 1],
        "actions": actions[:used].copy(),
        "frames_local": frames[: used + 1],
        "connected": connected[: used + 1].copy(),
        "panel_counts_truth": counts[: used + 1],
        "collisions_truth": collisions[:used],
        "attempted_position_truth": attempted[:used],
    }
    events = []
    results = []
    # Offline bookkeeping, deliberately outside native simulation/sensing loop.
    for e in range(n):
        memory = InspectionMemory(ticks + 1)
        truth_inspected = set()
        for k in range(used + 1):
            memory.observe(frames[k, e], k)
            accepted = evaluate_views(
                scene,
                states[k, e : e + 1, :3],
                states[k, e : e + 1, 6:10],
                counts[k, e : e + 1],
            )[0]
            truth_inspected.update(
                scene.evaluator_ids[j] for j in np.flatnonzero(accepted)
            )
        events.extend({**event, "env": e} for event in memory.events)
        for k in range(1, used + 1):
            if connected[k, e] != connected[k - 1, e]:
                events.append(
                    {
                        "tick": k,
                        "env": e,
                        "type": "link_restored" if connected[k, e] else "link_lost",
                    }
                )
        if terminal:
            events.append(
                {
                    "tick": used,
                    "env": e,
                    "type": "collision" if collisions[used - 1, e] else "batch_stopped",
                }
            )
        missed = sorted(set(scene.evaluator_ids) - truth_inspected)
        results.append(
            {
                "observed_discovered": sorted(memory.discovered),
                "observed_inspected": sorted(memory.inspected),
                "duplicate_views": memory.duplicate_views,
                "actor_coverage": "unknown",
                "termination": "collision_batch_stop" if terminal else memory.status,
                "evaluator_inspected_ids": sorted(truth_inspected),
                "evaluator_missed_ids": missed,
                "evaluator_complete": not missed,
            }
        )
    metadata = {
        "schema": "flightrl.inspection_replay.v2",
        "authority": "simulation_only",
        "deployment_authority": False,
        "policy": None,
        "controller": "open_loop_diagnostic",
        "scenario_sha256": scene.scenario.manifest["sha256"],
        "scene_sha256": scene.manifest["sha256"],
        "native_binary_sha256": sha256_file(_binding.__file__),
        "recorder_sha256": sha256_file(__file__),
        "camera": CAMERA,
        "state_fields": STATE_FIELDS,
        "dt_s": dt_s,
        "ticks": used,
        "num_envs": n,
        "panel_count": len(scene.panels),
        "requested_ticks": ticks,
        "quality": QUALITY,
        "state_source": "simulator_truth_evaluator_only_no_estimator",
        "link_model": "scripted_observed_status_no_rf_physics",
        "collision_response": "terminate_batch_restore_last_valid_state_record_attempt",
        "events": sorted(events, key=lambda event: (event["tick"], event["env"])),
        "results": results,
    }
    return metadata, arrays
