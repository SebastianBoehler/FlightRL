"""One MuJoCo owner for drone forces, rover wheel joints and industrial contacts."""

from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj
from .spec import import_robot
from .model_asset import attach_arm, ArticulatedRobot, ARM_SOURCE, asset_identity
from .render_geometry import geometry_description
from .drone_asset import attach_fpv, model_identity, fpv_source_identity

SOURCE = Path(__file__).resolve().parents[3] / "assets/robots/inspection_pair.xml"


class RobotWorld:
    dt = 0.002

    def __init__(self, equipment=(), site=None, arm=False):
        xml = ET.fromstring(SOURCE.read_text())
        attach_fpv(xml)
        world = xml.find("worldbody")
        self.site = site
        self.wind = np.array(site["wind"] if site else [0, 0, 0])
        if site:
            world.find("geom[@name='floor']").set("size", "120 100 .1")
            gate = ET.SubElement(
                world, "body", name="maintenance_gate", mocap="true", pos="0 0 5.2"
            )
            ET.SubElement(
                gate,
                "geom",
                name="maintenance_screen",
                type="box",
                size=".05 .65 .65",
                rgba=".8 .45 .05 1",
            )
        for box in equipment:
            ET.SubElement(
                world,
                "geom",
                name=box["name"],
                type=box.get("type", "box"),
                quat=" ".join(map(str, box.get("quaternion", [1, 0, 0, 0]))),
                pos=" ".join(map(str, box["position"])),
                size=" ".join(
                    map(str, box["size"] if "size" in box else box["half_extents"])
                ),
                rgba=" ".join(map(str, box["color"])),
                contype="1",
                conaffinity="1",
            )
        spec = mj.MjSpec.from_string(ET.tostring(xml, encoding="unicode"))
        if arm:
            attach_arm(spec)
        self.model = spec.compile()
        self.data = mj.MjData(self.model)
        self.arm = ArticulatedRobot(self.model, self.data) if arm else None
        self.drive_ids = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ("left_drive", "right_drive")
        ]
        self.free_joints = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, n)
            for n in ("drone_free", "rover_free")
        ]
        self.drone_dof = self.model.jnt_dofadr[self.free_joints[0]]
        if site:
            for joint, position in zip(self.free_joints, site["spawns"]):
                address = self.model.jnt_qposadr[joint]
                self.data.qpos[address : address + 3] = position
                self.data.qpos[address + 3 : address + 7] = [
                    np.cos(site["yaw"] / 2),
                    0,
                    0,
                    np.sin(site["yaw"] / 2),
                ]
            for i in range(self.model.ngeom):
                name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, i) or ""
                if name == "floor" or "tire" in name:
                    self.model.geom_friction[i, 0] = site["friction"]
        mj.mj_forward(self.model, self.data)
        self.specs = [
            import_robot(self.model, SOURCE, name) for name in ("drone", "rover")
        ]
        self.specs[0] = replace(self.specs[0], source_sha256=fpv_source_identity(SOURCE))
        if arm:
            self.specs.append(import_robot(self.model, ARM_SOURCE, "arm/link_base"))
            self.specs[-1] = replace(self.specs[-1], source_sha256=asset_identity())
        self.drone = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "drone")
        self.rover = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "rover")
        self.camera_names = ["drone_camera", "rover_camera"] + (
            ["arm/wrist_camera"] if arm else []
        )
        self.camera_sites = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, n)
            for n in self.camera_names
        ]
        self.camera_bodies = [int(self.model.site_bodyid[i]) for i in self.camera_sites]
        self.thrust = 1.0
        self.rate_command = np.zeros(3)
        self.actions = np.zeros(4)
        self.wheels = np.zeros(2)
        self.contact_events = 0
        self.robot_collision_steps = 0
        self.robot_geoms = set()
        self.support_geoms = set()
        for geom in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[geom])
            while body and body not in (self.drone, self.rover):
                body = int(self.model.body_parentid[body])
            if body:
                self.robot_geoms.add(geom)
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, geom) or ""
            if name in ("floor", "dock_platform") or name.startswith("lane_paint"):
                self.support_geoms.add(geom)
        self.ticks = 0

    def step(self):
        if self.site:
            y = self.site["spawns"][0][1]
            self.data.mocap_pos[0] = [
                0.1,
                y,
                1.8 if 4 < self.ticks * self.dt < 8 else 5.2,
            ]
        if not np.isfinite(self.actions).all() or not np.isfinite(self.wheels).all():
            raise ValueError("Finite actuator commands required")
        rotation = self.data.xmat[self.drone].reshape(3, 3)
        alpha = self.dt / (0.04 + self.dt)
        self.thrust += alpha * (
            1 + 0.75 * np.clip(self.actions[0], -1, 1) - self.thrust
        )
        self.rate_command += (
            self.dt
            / (0.08 + self.dt)
            * (np.clip(self.actions[1:], -1, 1) * [2, 2, 1.5] - self.rate_command)
        )
        rates = self.data.qvel[self.drone_dof + 3 : self.drone_dof + 6]
        inertia = self.model.body_inertia[self.drone]
        torque = inertia * (self.rate_command - rates) / 0.04 + np.cross(
            rates, inertia * rates
        )
        self.data.xfrc_applied[:] = 0
        self.data.xfrc_applied[self.drone, :3] = rotation[:, 2] * self.model.body_mass[
            self.drone
        ] * 9.81 * self.thrust - 0.1 * self.model.body_mass[self.drone] * (
            self.data.qvel[self.drone_dof : self.drone_dof + 3] - self.wind
        )
        self.data.xfrc_applied[self.drone, 3:] = rotation @ np.clip(torque, -0.04, 0.04)
        self.data.ctrl[self.drive_ids] = np.clip(self.wheels, -6, 6)
        if self.arm:
            self.arm.apply()
        mj.mj_step(self.model, self.data)
        self.ticks += 1
        self.contact_events += int(self.data.ncon > 0)
        for c in self.data.contact[: self.data.ncon]:
            pair = {int(c.geom1), int(c.geom2)}
            if pair & self.robot_geoms and not pair & self.support_geoms:
                self.robot_collision_steps += 1
                break

    def bodies(self):
        return dict(
            positions=self.data.xpos[1:].tolist(),
            quaternions=self.data.xquat[1:][:, [1, 2, 3, 0]].tolist(),
        )

    def cameras(self):
        # Convert the physical camera mount into the shared renderer's body-offset contract.
        positions = []
        quaternions = []
        for site in self.camera_sites:
            r = self.data.site_xmat[site].reshape(3, 3)
            q = np.zeros(4)
            mj.mju_mat2Quat(q, r.ravel())
            positions.append(
                (self.data.site_xpos[site] - r @ np.array([0.035, 0, 0.012])).tolist()
            )
            quaternions.append(q[[1, 2, 3, 0]].tolist())
        return dict(positions=positions, quaternions=quaternions)

    def render_description(self):
        return dict(
            **geometry_description(self.model),
            cameras=[
                dict(id=n, body=b, robot_id=r)
                for n, b, r in zip(
                    self.camera_names, self.camera_bodies, ("drone", "rover", "arm")
                )
            ],
            robots=[s.record() for s in self.specs],
            drone_reference=model_identity("fpv"),
            body_count=self.model.nbody - 1,
            physics_dt_s=self.dt,
            frames=dict(
                world="Z up",
                body="X forward, Y left, Z up",
                position="m",
                quaternion="xyzw",
            ),
            actuation=dict(
                drone="Normalized collective and body rates; collective 0.25-1.75 times weight, rate limits 2/2/1.5 rad/s, torque limit 0.04 Nm per axis; 40ms thrust and 80ms rate lag",
                rover="Wheel velocity rad/s, 0.11m radius, 0.42m track, +/-6 rad/s and +/-3 Nm",
            ),
        )
