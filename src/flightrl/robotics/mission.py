"""Image-based work orders, estimated inspection handovers and separate truth scoring."""

import hashlib
import mujoco as mj
import numpy as np
from flightrl import _binding
from flightrl.fleet.vehicles import VEHICLES
from .sensing import Observation, servo, target_from_pixels
from .visual_control import inspect_marker, features, teacher, supervise


class InspectionMission:
    def __init__(self, world, targets, policy=None):
        self.world = world
        self.targets = {t["id"]: t for t in targets}
        self.target_ids = [t["id"] for t in targets]
        self.industry = world.site is not None
        self.policy = policy
        self.phase = ["inspect", "await_task" if self.industry else "inspect"]
        self.dwell = [0, 0]
        self.events = []
        self.correct = {}
        self.evidence = []
        self.report_sent = None
        self.report_received = None
        self.handover = None
        self.link = True
        self.commands = np.zeros((2, 4), np.float32)
        self.observations = 0
        self.last_seen = [None, None]
        self.observed_time = 0.0
        self.last_visible = [-100.0, -100.0]
        self.active_targets = [None, None]
        self.control_estimate = None
        self.feature_log = []
        self.observation_sequence = None

    def observe(self, frames, state):
        now = state["time_s"]
        self.observed_time = now
        self.observation_sequence = state["sequence"]
        self.observations += 1
        if (
            self.link
            and self.report_received is None
            and self.report_sent is not None
            and now >= self.report_sent + 0.2
        ):
            self.report_received = now
            if self.phase[1] == "await_task":
                self.phase[1] = "inspect"
        for i, kind in enumerate(("drone", "rover")):
            target = (
                self.target_ids[0]
                if i == 0
                else (42 if self.phase[i] == "dock" else self.target_ids[1])
            )
            if self.industry and i == 1 and self.phase[i] == "inspect":
                target = self.handover["followup_marker"]
                if target not in self.targets:
                    raise ValueError("Handover names an unknown work-order marker")
            if target != self.active_targets[i]:
                self.last_visible[i] = -100.0
                self.active_targets[i] = target
            rgb, depth = frames[i][0]
            valid = np.isfinite(depth).mean() > 0.5
            measurement = (
                inspect_marker(rgb, depth, target) if self.industry and valid else None
            )
            relative = (
                (None if measurement is None else measurement["relative"])
                if self.industry
                else target_from_pixels(rgb, depth, target)
            )
            self.last_seen[i] = None if relative is None else relative.tolist()
            command, found = servo(relative, kind)
            if self.industry:
                observation = features(
                    measurement, depth, np.array(state["proprio"][i], np.float32), kind
                )
                reference, found = teacher(measurement, depth, kind)
                command = reference
                if self.policy is not None:
                    if not getattr(self.policy, "controls_both", False):
                        raise ValueError(
                            "Industrial work orders require the mixed-robot feature policy"
                        )
                    command = self.policy.act(observation)
                command = supervise(command, depth, kind, valid)
                if relative is not None:
                    self.last_visible[i] = now
                elif now - self.last_visible[i] < 5:
                    command[:] = 0
                if target != 42:
                    found = (
                        found
                        and measurement is not None
                        and measurement["signal"] is not None
                    )
                self.feature_log.append(
                    dict(
                        features=observation.copy(),
                        action=reference.copy(),
                        robot=i,
                        time_s=now,
                    )
                )
            elif i == 0 and self.policy is not None and self.phase[i] == "inspect":
                command, confidence = self.policy(
                    Observation(
                        *frames[i][1],
                        np.array(state["proprio"][i], np.float32),
                        state["sequence"],
                        now,
                    )
                )
                found = confidence > 0.8
            if self.phase[i] == "hold" and self.industry and i == 0:
                found = False  # Keep visual station-keeping active with biased motion estimates.
            elif self.phase[i] in ("hold", "done", "wait", "await_task"):
                command[:] = 0
                found = False
            self.commands[i] = command
            self.dwell[i] = self.dwell[i] + 1 if found and valid else 0
            if self.dwell[i] >= 4:
                self.record(i, kind, target, measurement, state, rgb)
                self.dwell[i] = 0
                if i == 0:
                    self.phase[i] = "hold"
                    self.report_sent = now
                elif target == self.target_ids[1]:
                    self.phase[i] = "wait"
                else:
                    self.phase[i] = "done"
        if self.phase[1] == "wait" and self.report_received is not None:
            self.phase[1] = "dock"

    def record(self, i, kind, target, measurement, state, rgb):
        # This evaluator is the only mission operation that reads the authored target position.
        p = np.array(state["camera_poses"][i]["position_m"])
        goal = np.array(self.targets[target]["position"])
        stand = 1.05 if i == 0 else 0.55
        expected = goal + stand * np.array(
            self.targets[target].get(
                "approach", [1, 0, 0] if target == 42 else [-1, 0, 0]
            )
        )
        if i == 1:
            expected[2] = p[2]
        error = float(np.linalg.norm(p - expected))
        observed_signal = None if measurement is None else measurement["signal"]
        signal_correct = (
            not self.industry or observed_signal == self.targets[target]["signal"]
        )
        correct = error < (0.25 if i == 0 else 0.22) and signal_correct
        self.correct[target] = correct
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        self.events.append(
            dict(
                time_s=state["time_s"],
                capture_sequence=state["sequence"],
                decision_time_s=self.world.ticks * self.world.dt,
                robot=kind,
                target=target,
                verified=correct,
                camera_position_m=p.tolist(),
                position_error_m=error,
                observed_signal=observed_signal,
                signal_correct=signal_correct,
                image_sha256=digest,
            )
        )
        self.evidence.append(rgb.copy())
        if self.industry and i == 0:
            estimate = state["estimation"]
            rotation = np.zeros(9)
            mj.mju_quat2Mat(rotation, np.array(estimate["quaternions"][i]))
            position = np.array(estimate["positions"][i]) + rotation.reshape(3, 3) @ (
                measurement["relative"] + [0.035, 0, 0.012]
            )
            self.handover = dict(
                asset_id=target,
                followup_marker=target + 6,
                observed_signal=observed_signal,
                estimated_position_m=position.tolist(),
                position_variance_m2=float(estimate["variance"][i] + 0.02**2),
                capture_time_s=state["time_s"],
                image_sha256=digest,
            )

    def apply(self):
        world = self.world
        if world.ticks * world.dt - self.observed_time > 0.3:
            self.commands[:] = 0
        command = self.commands[0]
        if self.control_estimate is None:
            q = world.data.xquat[world.drone][None].astype(np.float32)
            v = world.data.qvel[world.drone_dof:world.drone_dof + 3][None].astype(np.float32)
        else:
            q = self.control_estimate.q[:1]
            v = self.control_estimate.velocity[:1]
        normalized = np.array(
            [[command[0] / 0.7, command[1] / 0.7, command[2] / 0.4, command[3] / 2.5]],
            np.float32,
        )
        out = np.empty((1, 4), np.float32)
        _binding.sixdof_setpoint_actions(
            v,
            q,
            normalized,
            VEHICLES["fpv"].physics()[None],
            out,
            0.7,
            0.4,
            2.5,
            6.0,
            3.0,
        )
        world.actions[:] = out[0]
        forward, yaw = self.commands[1, 0], self.commands[1, 3]
        world.wheels[:] = [(forward - yaw * 0.21) / 0.11, (forward + yaw * 0.21) / 0.11]

    def status(self):
        label = (
            "learned drone + rover"
            if getattr(self.policy, "controls_both", False)
            else "learned drone + visual rover"
            if self.policy is not None
            else "visual baselines"
        )
        return dict(
            phases=self.phase,
            events=self.events,
            success=all(self.correct.get(t, False) for t in self.target_ids)
            and self.world.robot_collision_steps == 0,
            link=self.link,
            report_delivered=self.report_received is not None,
            seen=self.last_seen,
            handover=self.handover,
            controller=label,
        )
