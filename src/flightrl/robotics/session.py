"""Live mixed robot episode, sensor clocks, observations and reproducible reports."""

import hashlib
import json
import time
from threading import Lock
import numpy as np
from .environment import RobotEnvironment
from .sensing import decode
from .depth_audit import depth_audit
from .recording import RunRecorder


class RobotSession:
    def __init__(self, folder, seed=0, policy=None, industry=False, arm=False):
        folder.mkdir(parents=True, exist_ok=False)
        self.folder = folder
        self.started = time.perf_counter()
        self.save_lock = Lock()
        self.stop_reason = None
        self.link_events = []
        self.industry = industry
        self.environment = RobotEnvironment(seed, policy, industry, arm)
        self.world = self.environment.world
        self.mission = self.environment.mission
        self.sensor_rig = self.environment.sensor_rig
        self.targets = self.environment.targets
        self.pending = None
        self.meta = None
        self.paused = False
        self.count = 0
        self.timings = []
        self.physics = []
        self.display = []
        self.samples = []
        self.trace = []
        self.depth_checks = []
        self.capture_index = []
        description = self.environment.description()
        self.description = description
        self.identity = dict(
            schema_version=2,
            episode_id=folder.name,
            clock_id="simulation",
            robot_sources=[s.source_sha256 for s in self.world.specs],
            arm_controller="operator joint setpoints" if arm else None,
            seed=seed,
            scene_sha256=hashlib.sha256(
                json.dumps(description, sort_keys=True).encode()
            ).hexdigest(),
            scope="Simulation only. Surrogate robots; modeled sensor errors and wheel odometry."
            if industry
            else "Simulation only. Surrogate drone and rover; ideal proprioception, actual joint contacts and rendered camera observations.",
            sensor_contract=(
                "512x384 RGB and metric ray depth feed explicit AprilTag/plane/indicator perception; learned actor receives 21 marker, depth-clearance, proprioception and robot-role features. 63 degree vertical FOV, max 8m."
                if industry
                else "RGB and metric ray-distance depth, 512x384 perception, 128x96 learned actor; 63 degree vertical FOV, max 8m."
            ),
            controller=self.mission.status()["controller"],
            actor_sha256=policy.sha256 if policy else None,
            physics_engine="MuJoCo " + __import__("mujoco").__version__,
        )
        (folder / "scene.json").write_text(json.dumps(description, indent=2))
        (folder / "identity.json").write_text(json.dumps(self.identity, indent=2))
        self.recorder = RunRecorder(folder, self.identity, description)

    def state(self):
        return self.environment.state()

    def step(self):
        start = time.perf_counter()
        application_tick = self.world.ticks
        feedback = self.state()
        self.environment.step()
        stamp = round(application_tick * self.world.dt * 1e9)
        self.recorder.submit(
            "/actuation",
            dict(
                application_tick=application_tick,
                application_time_ns=stamp,
                feedback_proprio=feedback["proprio"],
                feedback_estimation=feedback.get("estimation"),
                feedback_tick=application_tick,
                source_capture_sequence=self.mission.observation_sequence,
                source_capture_time_s=self.mission.observed_time
                if self.mission.observation_sequence is not None
                else None,
                drone=self.world.actions.tolist(),
                rover=self.world.wheels.tolist(),
                arm=self.world.arm.target.tolist() if self.world.arm else None,
            ),
            stamp,
            stamp,
            application_tick,
        )
        state = self.state()
        self.recorder.submit(
            "/state",
            state,
            state["capture_time_ns"],
            state["capture_time_ns"],
            state["sequence"],
        )
        self.physics.append((time.perf_counter() - start) * 1000)
        if self.industry and self.world.robot_collision_steps:
            self.paused = True
            self.stop_reason = "Robot contacted industrial equipment"
        if self.world.ticks % 50 == 0:
            self.trace.append(
                dict(
                    time_s=self.world.ticks * self.world.dt,
                    drone=self.world.data.xpos[self.world.drone].tolist(),
                    rover=self.world.data.xpos[self.world.rover].tolist(),
                )
            )
        if self.world.data.xpos[
            self.world.drone, 2
        ] > 5 or self.world.ticks * self.world.dt >= (180 if self.industry else 120):
            self.paused = True
            self.stop_reason = "Episode time or altitude limit reached"

    def receive(self, data):
        if (
            self.pending is None
            or self.meta is None
            or self.meta["id"] != self.pending[0]
        ):
            raise ValueError("Unexpected or mismatched sensor batch")
        request, state, started = self.pending
        frames = decode(data, len(self.world.camera_sites))
        if self.industry and self.count in (0, 50, 100):
            self.depth_checks.append(depth_audit(self.world, frames, state))
        self.environment.observe(frames, state)
        available_ns = round(self.world.ticks * self.world.dt * 1e9)
        state = {
            **state,
            "available_time_ns": available_ns,
            "episode_id": self.folder.name,
            "received_monotonic_ns": time.monotonic_ns(),
        }
        self.recorder.submit(
            "/capture", state, state["capture_time_ns"], available_ns, state["sequence"]
        )
        self.recorder.frames("raw", frames, state, available_ns)
        self.capture_index.append(
            dict(
                sequence=state["sequence"],
                time_s=state["time_s"],
                available_time_ns=available_ns,
            )
        )
        delivery = self.environment.last_delivery
        if delivery is not None:
            observed, observation = delivery
            self.recorder.frames("observed", observed, observation, available_ns)
            self.recorder.submit(
                "/decision",
                dict(
                    capture_time_ns=observation["capture_time_ns"],
                    capture_sequence=observation["sequence"],
                    decision_tick=self.world.ticks,
                    decision_time_ns=available_ns,
                    commands=self.mission.commands.tolist(),
                    proprio=observation["proprio"],
                    events=list(self.mission.events),
                ),
                observation["capture_time_ns"],
                available_ns,
                observation["sequence"],
            )
        self.count += 1
        self.timings.append((time.perf_counter() - started) * 1000)
        if delivery is not None and len(self.samples) < 150:
            self.samples.append(
                dict(
                    rgb=observed[0][1][0],
                    depth=observed[0][1][1],
                    proprio=np.array(observation["proprio"][0], np.float32),
                    commands=self.mission.commands[0].copy(),
                    sequence=observation["sequence"],
                    time_s=observation["time_s"],
                    decision_time_ns=available_ns,
                )
            )
        self.pending = None
        self.meta = None
        if self.mission.status()["success"]:
            self.paused = True
        elif len(self.mission.events) == 3:
            self.paused = True
            self.stop_reason = (
                "Mission rejected by independent inspection or collision check"
            )
        status = self.mission.status()
        text = f"{status['controller']} · drone: {status['phases'][0]} · rover: {status['phases'][1]} · report {'delivered' if status['report_delivered'] else 'pending'} · {len(status['events'])}/3 reports"
        if self.stop_reason:
            text += " · " + self.stop_reason
        if status["success"]:
            text += " · all inspections and docking verified"
        return dict(
            type="metrics",
            count=self.count,
            status=text,
            done=self.paused,
            handover=status["handover"],
            sensor_valid=self.sensor_rig.valid if self.sensor_rig else [True, True],
            observation_sequence=self.mission.observation_sequence,
            observation_age_s=self.world.ticks * self.world.dt
            - self.mission.observed_time,
        )

    def save(self):
        self.recorder.finish()

        def stats(values):
            return (
                {
                    "median": float(np.median(values)),
                    "p95": float(np.quantile(values, 0.95)),
                }
                if values
                else None
            )

        report = {
            **self.identity,
            **self.mission.status(),
            "time_s": self.world.ticks * self.world.dt,
            "wall_s": time.perf_counter() - self.started,
            "camera_batches": self.count,
            "camera_batch_ms": stats(self.timings),
            "physics_20ms_ms": stats(self.physics),
            "display": self.display,
            "trace": self.trace,
            "stop_reason": self.stop_reason,
            "link_events": self.link_events,
            "physics_steps_with_contact": self.world.contact_events,
            "robot_collision_steps": self.world.robot_collision_steps,
            "render_physics_depth_checks": self.depth_checks,
            "sensors": self.sensor_rig.record() if self.sensor_rig else None,
            "capture_index": self.capture_index,
            "recording": "run.mcap",
            "recorded_messages": self.recorder.count,
            "recording_queue_peak": self.recorder.queue_peak,
        }
        with self.save_lock:
            if self.mission.evidence:
                import cv2

                for i, rgb in enumerate(self.mission.evidence):
                    cv2.imwrite(
                        str(self.folder / f"inspection-{i}.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    )
            if self.mission.feature_log:
                np.savez_compressed(
                    self.folder / "features.npz",
                    **{
                        k: np.array([r[k] for r in self.mission.feature_log])
                        for k in self.mission.feature_log[0]
                    },
                )
            temporary = self.folder / "report.json.tmp"
            temporary.write_text(json.dumps(report, indent=2))
            temporary.replace(self.folder / "report.json")
            if self.samples:
                temporary = self.folder / "observations.npz.tmp"
                with temporary.open("wb") as output:
                    np.savez_compressed(
                        output,
                        **{
                            k: np.array([s[k] for s in self.samples])
                            for k in self.samples[0]
                        },
                    )
                temporary.replace(self.folder / "observations.npz")
