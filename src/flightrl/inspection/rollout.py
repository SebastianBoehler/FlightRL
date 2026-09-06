"""Native closed-loop RGB-D mission runner with modeled inertial odometry."""

from time import perf_counter
import numpy as np
from flightrl import _binding
from flightrl.artifact_identity import sha256_file
from flightrl.inspection.controller import MissionController
from flightrl.inspection_scene import swept_collision
from flightrl.inspection_mission import evaluate_views
from flightrl.sixdof.native import native_step


def run_mission(
    scene,
    *,
    ticks=1200,
    link_loss=False,
    policy=None,
    seed=0,
    collect=False,
    fault=None,
    industrial=False,
    controller_factory=None,
    sensor_size=(256, 192),
):
    if fault not in (None, "blocked_route", "continued_outage", "estimator_loss"):
        raise ValueError("unknown failure injection")
    w, h = sensor_size if industrial else (64, 48)
    if (w, h) not in ((64, 48), (128, 96), (256, 192), (512, 384), (768, 576)):
        raise ValueError("unsupported sensor_size")
    factor = w // 64
    rng = np.random.default_rng(seed)
    dt = 0.1
    p = np.array([[-2, -1.5, 1.5]], np.float32)
    v = np.zeros((1, 3), np.float32)
    q = np.array([[1, 0, 0, 0]], np.float32)
    rates = v.copy()
    thrust = np.ones(1, np.float32)
    physics = scene.scenario.arrays["vehicle_physics"][None, :].copy()
    ranges = np.empty((1, 6), np.float32)
    rgb = np.zeros((1, 48, 64, 3), np.uint8)
    depth = np.zeros((1, 48, 64), np.float32)
    counts = np.zeros((1, len(scene.panels), 2), np.int32)
    action = np.zeros((1, 4), np.float32)
    estimate = p[0].copy()
    from flightrl.inspection.industrial import IndustrialMission
    from flightrl.inspection.conditions import PlantConditions

    controller = (
        controller_factory or (IndustrialMission if industrial else MissionController)
    )(estimate)
    conditions = PlantConditions(seed, scene, (w, h)) if industrial else None
    if conditions:
        physics[0, 2] = conditions.profile.air_drag_per_s
    sensor_rgb = np.zeros((1, h, w, 3), np.uint8) if industrial else rgb
    sensor_depth = np.zeros((1, h, w), np.float32) if industrial else depth
    hidden = None
    records = []
    frames = []
    depths = []
    dataset = []
    truth_inspected = set()
    collision = False
    physics_steps = 0
    start = perf_counter()
    link_connected = True
    loss_tick = None
    active_scene = scene
    for tick in range(ticks):
        if fault == "blocked_route" and loss_tick is not None and tick == loss_tick + 5:
            from flightrl.scenario_bundle import CompiledScenarioBundle
            from dataclasses import replace

            arrays = dict(scene.scenario.arrays)
            barrier = np.array([[0.55, 0.85, -1.8, -0.7, 0, 2.5]], np.float32)
            arrays["terrain_obstacles"] = np.concatenate(
                (arrays["terrain_obstacles"], barrier)
            )
            active_scene = replace(
                scene, scenario=CompiledScenarioBundle(scene.scenario.manifest, arrays)
            )
            if conditions:
                conditions.set_obstacles(arrays["terrain_obstacles"])
            controller.events.append(
                {"tick": tick, "type": "obstacle_injected", "boxes": barrier.tolist()}
            )
        _binding.inspection_render(
            p,
            q,
            scene.scenario.arrays["terrain_bounds"],
            active_scene.scenario.arrays["terrain_obstacles"],
            scene.panels,
            sensor_rgb,
            counts,
            sensor_depth,
            int(industrial),
            *(conditions.render_buffers if conditions else ()),
        )
        if conditions:
            conditions.camera(sensor_rgb[0], sensor_depth[0], p[0], q[0])
            conditions.optics.apply(sensor_rgb[0])
            rgb[:] = (
                sensor_rgb.reshape(1, 48, factor, 64, factor, 3)
                .mean(axis=(2, 4))
                .astype(np.uint8)
            )
            depth[:] = sensor_depth[:, factor // 2 :: factor, factor // 2 :: factor]
            counts //= factor * factor
        # Sensor model: measured world-frame velocity + small bias/noise, known takeoff origin.
        # Not VIO and never injects simulator position corrections after initialization.
        measured_velocity = (
            v[0] + np.array([0.0005, -0.0003, 0]) + rng.normal(0, 0.001, 3)
        )
        if tick:
            estimate += measured_velocity * dt
        if link_loss:
            if link_connected and p[0, 0] >= 1.15:
                link_connected = False
                loss_tick = tick
            elif not link_connected and p[0, 0] < 0.65 and fault != "continued_outage":
                link_connected = True
        connected = link_connected
        if fault == "estimator_loss" and tick == 200:
            controller.finished = True
            controller.mode = "localization_lost"
            controller.events.append({"tick": tick, "type": "localization_lost"})
        controller.observe(rgb[0], depth[0], estimate, q[0], connected)
        command, goal = controller.command(estimate, q[0])
        teacher = command.copy()
        # Local sensing brake used by both methods and recovery (camera faces travel).
        clearance = float(np.quantile(depth[0, 18:30, 26:38], 0.1))
        if teacher[0] > 0:
            teacher[0] *= np.clip((clearance - 0.35) / 0.6, 0, 1)
        proprio = np.r_[measured_velocity, q[0], goal].astype(np.float32)
        if collect:
            dataset.append((rgb[0].copy(), depth[0].copy(), proprio, teacher.copy()))
        if policy is not None:
            command, hidden = policy(rgb[0], depth[0], proprio, hidden)
            command = np.asarray(command, np.float32)
            # Shared explicit safety and altitude supervisor, not a teacher fallback.
            if command[0] > 0:
                command[0] *= np.clip((clearance - 0.35) / 0.6, 0, 1)
            command[2] = teacher[2]
            if controller.mode == "scan":
                command[:2] = teacher[:2]
                command[3] = teacher[3]
        else:
            command = teacher
        accepted = evaluate_views(scene, p, q, counts)[0]
        truth_inspected.update(scene.evaluator_ids[j] for j in np.flatnonzero(accepted))
        records.append(
            {
                "tick": tick,
                "time_s": round(tick * dt, 3),
                "position": p[0].tolist(),
                "quaternion": q[0].tolist(),
                "estimate": estimate.tolist(),
                "connected": bool(connected),
                "mode": controller.mode,
                "inspected": sorted(controller.inspected),
                "discovered": sorted(controller.panels),
                "truth_inspected": sorted(truth_inspected),
                "command": command.tolist(),
                "clearance": clearance,
            }
        )
        if conditions:
            records[-1].update(conditions.record())
        frames.append(sensor_rgb[0].copy())
        depths.append(depth[0].copy())
        if controller.finished:
            break
        _binding.sixdof_setpoint_actions(
            v, q, command[None, :].copy(), physics, action, 0.65, 0.4, 2.5, 6.0, 3.0
        )
        for _ in range(5):
            before = p.copy()
            if conditions:
                conditions.step(v, 0.02, p[0], q[0], float(thrust[0]))
            native_step(
                p,
                v,
                q,
                rates,
                ranges,
                action,
                0.02,
                scene.scenario.arrays["terrain_bounds"],
                thrust,
                physics,
            )
            physics_steps += 1
            if swept_collision(active_scene, before, p).any():
                collision = True
                break
        if collision:
            break
    status = (
        "collision"
        if collision
        else controller.mode
        if controller.finished
        else "budget_exhausted"
    )
    result = {
        "status": status,
        "environment": scene.environment.report() if industrial else None,
        "scene_sha256": scene.manifest["sha256"],
        "native_binary_sha256": sha256_file(_binding.__file__),
        "rollout_source_sha256": sha256_file(__file__),
        "ticks": len(records),
        "simulated_s": round(physics_steps * 0.02, 2),
        "last_capture_time_s": records[-1]["time_s"],
        "terminal_position": p[0].tolist(),
        "wall_s": perf_counter() - start,
        "inspected": sorted(truth_inspected),
        "missed": sorted(set(scene.evaluator_ids) - truth_inspected),
        "coverage": len(truth_inspected) / max(1, len(scene.panels)),
        "collision": collision,
        "recovered": controller.recovered,
        "operator_frames_withheld": sum(not r["connected"] for r in records),
        "odometry_final_error_m": float(np.linalg.norm(estimate - p[0])),
        "events": controller.events,
        "controller": getattr(controller, "label", "classical_rgbd")
        if policy is None
        else "recurrent_visual_student",
    }
    return result, records, np.array(frames), np.array(depths), dataset
