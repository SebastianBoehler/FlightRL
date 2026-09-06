"""One local simulation session, frozen sensor actor and empty-start reconstruction."""

import json
import time
import threading
import hashlib
from pathlib import Path
import numpy as np
from flightrl.fleet.camera_policy.network import Policy
from flightrl.reconstruction.geometry import intrinsics
from flightrl.reconstruction.odometry import VisualOdometry
from flightrl.reconstruction.fusion import SurfaceMap
from .physics import ContactWorld
from .particles import Particles
from .demo import DemoFlight
from .sensors import decode_frames, actor_packet
from flightrl.robotics.drone_asset import model_identity


class Session:
    def __init__(self, payload, actor, folder):
        folder.mkdir(parents=True, exist_ok=False)
        self.folder = folder
        self.save_lock = threading.Lock()
        self.world = ContactWorld(payload)
        self.particles = Particles(self.world)
        self.policy = Policy(actor)
        self.demo = DemoFlight(self.world)
        self.mode = "hover"
        self.notice = ""
        self.started = time.perf_counter()
        self.k = intrinsics()
        self.vo = [VisualOdometry(self.k, "rgbd") for _ in range(3)]
        self.maps = [SurfaceMap() for _ in range(3)]
        self.samples = []
        self.latencies = []
        self.steps = []
        self.sensor_count = 0
        self.display = []
        self.reported_at = np.full(3, np.inf)
        self.dwell = np.zeros(3, int)
        self.mode_samples = {"hover": 0, "dust": 0, "policy": 0, "paused": 0}
        self.pending = None
        self.camera_meta = None
        root = Path(__file__).resolve().parents[3]
        sources = sorted(
            [
                *root.glob("src/flightrl/realism/*.py"),
                *root.glob("src/flightrl/native/realism/*"),
                *root.glob("viewer/src/realism/*"),
                *root.glob("viewer/src/forest/*.ts"),
                *root.glob("viewer/src/models/*.ts"),
                *root.glob("assets/robots/drone_models/*.json"),
                root / "scripts/run_realism.py",
            ]
        )
        hashes = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
            if p.is_file()
        }
        self.identity = {
            "source_sha256": hashes,
            "scene_sha256": self.world.scene_hash,
            "actor_sha256": hashlib.sha256(actor.read_bytes()).hexdigest(),
            "geometry": "Exact solid triangles exported from the displayed forest; soft foliage has no rigid collider",
            "depth": "Ray distance metres, max/no-hit 8m, top-left origin, research vertical FOV 63deg; body offsets declared in drone_references",
            "drone_references": [model_identity(b["vehicle"]) for b in self.world.specs[:3]],
            "actor": "Same scene rendered independently at 64x48; high-resolution 256x192 for reconstruction",
            "physics": "Jolt 5.3.0, 20ms, 1mm contact tolerance; native actuator response; box inertia from explicit mass and dimensions",
            "scope": "Simulation only; frozen actor not trained on this forest; reports are not mission success",
        }
        (folder / "scene.json").write_text(json.dumps(payload))
        (folder / "identity.json").write_text(json.dumps(self.identity, indent=2))

    def set_mode(self, mode):
        if mode not in ("paused", "hover", "dust", "policy"):
            raise ValueError("Invalid simulation mode")
        if mode == "policy" and any(b["vehicle"] != "fpv" for b in self.world.specs[:3]):
            raise ValueError("The frozen FPV policy cannot control agricultural references; use Hold position or Dust demonstration")
        self.mode = mode
        self.notice = ""
        if mode == "dust":
            self.demo.started = self.world.ticks * self.world.dt
        if mode != "policy":
            self.world.actions[:] = 0

    def step(self):
        start = time.perf_counter()
        if self.mode in ("hover", "dust"):
            self.world.actions[:] = self.demo.controls(dust=self.mode == "dust")
        self.world.step()
        self.particles.step()
        if self.mode == "policy" and (
            (self.world.p[:, 2] > 6).any()
            or (self.world.p[:, 2] < -0.5).any()
            or (np.linalg.norm(self.world.p - self.demo.home, axis=1) > 12).any()
        ):
            self.mode = "paused"
            self.world.actions[:] = 0
            self.notice = "Experimental policy paused outside the demonstration volume. It has not been trained for this scene."
        self.steps.append((time.perf_counter() - start) * 1000)

    def state(self, particles=False):
        state = self.world.state(self.mode)
        state["notice"] = self.notice
        if particles:
            state.update(self.particles.record())
        return state

    def receive(self, data):
        if self.pending is None or self.camera_meta is None:
            raise ValueError("Unrequested camera frame")
        request, state, started = self.pending
        if self.camera_meta["id"] != request:
            raise ValueError("Camera sequence mismatch")
        frames = decode_frames(data)
        delivered = state["time_s"] >= self.reported_at[:2] + 0.2
        messages = np.tile(np.r_[delivered, [1, 1], [0.2, 0.2]], (3, 1)).astype(
            np.float32
        )
        packet = actor_packet(frames, state, messages)
        packet.validate()
        action, confidence = self.policy(packet)
        self.dwell = np.where(confidence > 0.8, self.dwell + 1, 0)
        new_report = (self.dwell >= 3) & ~np.isfinite(self.reported_at)
        self.reported_at[new_report] = state["time_s"]
        if self.mode == "policy":
            self.world.actions[:] = action
        tracked = []
        for i in range(3):
            rgb, depth = frames[i][0]
            pose = self.vo[i].step(rgb, depth)
            if pose is not None:
                self.maps[i].integrate(rgb, depth, self.k, pose, request)
            tracked.append(self.vo[i].status)
        elapsed = (time.perf_counter() - started) * 1000
        self.latencies.append(elapsed)
        self.sensor_count += 1
        if self.mode_samples[self.mode] < 50:
            self.mode_samples[self.mode] += 1
            self.samples.append(
                {
                    "rgb": packet.rgb,
                    "depth": packet.depth,
                    "proprio": packet.proprio,
                    "action": action,
                    "applied_action": self.world.actions.copy(),
                    "applied_time_s": self.world.ticks * self.world.dt,
                    "role": packet.role,
                    "messages": packet.messages,
                    "mode": self.mode,
                    "sequence": packet.sequence,
                    "time_s": packet.capture_time_s,
                    "high_rgb": np.stack([x[0][0] for x in frames]),
                    "high_depth": np.stack([x[0][1] for x in frames]),
                }
            )
        self.pending = None
        self.camera_meta = None
        return {
            "type": "metrics",
            "camera_batch_ms": round(elapsed, 2),
            "camera_batches": self.sensor_count,
            "tracked": tracked,
            "points": [len(m.voxels) for m in self.maps],
            "confidence": confidence.tolist(),
            "physics_p95_ms": round(float(np.quantile(self.steps[-500:], 0.95)), 2)
            if self.steps
            else 0,
            "scene_sha256": self.world.scene_hash,
            **self.particles.record(),
        }

    def save(self):
        with self.save_lock:
            self._save()

    def _save(self):
        def stats(a):
            return (
                {"p50": float(np.median(a)), "p95": float(np.quantile(a, 0.95))}
                if a
                else None
            )

        report = {
            **self.identity,
            "sensor_batches": self.sensor_count,
            "physics_steps": self.world.ticks,
            "physics_ms": stats(self.steps),
            "camera_batch_ms": stats(self.latencies),
            "display": self.display,
            "contacts": self.world.total_contacts,
            "recorded_batches": len(self.samples),
            "recording_limit": "First 50 batches per mode",
            "particle_counts": {
                k: v
                for k, v in self.particles.record().items()
                if k not in ("particles", "particleKinds")
            },
        }
        temporary = self.folder / "report.tmp.json"
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(self.folder / "report.json")
        if self.samples:
            keys = [
                "rgb",
                "depth",
                "proprio",
                "action",
                "sequence",
                "time_s",
                "high_rgb",
                "high_depth",
                "mode",
                "role",
                "messages",
                "applied_action",
                "applied_time_s",
            ]
            temporary = self.folder / "observations.tmp.npz"
            np.savez_compressed(
                temporary, **{k: np.array([s[k] for s in self.samples]) for k in keys}
            )
            temporary.replace(self.folder / "observations.npz")
