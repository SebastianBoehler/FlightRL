"""Attach a pinned mechanical model; MuJoCo owns joints, tendons and constraints."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import mujoco as mj
import numpy as np

ARM_SOURCE = Path(__file__).resolve().parents[3] / "assets/robots/xarm7/xarm7.xml"


@lru_cache(maxsize=1)
def asset_identity():
    manifest = json.loads((ARM_SOURCE.parent / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if (
            hashlib.sha256((ARM_SOURCE.parent / name).read_bytes()).hexdigest()
            != digest
        ):
            raise ValueError(f"Robot asset hash mismatch: {name}")
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def attach_arm(spec):
    asset_identity()
    child = mj.MjSpec.from_file(str(ARM_SOURCE))
    child.body("link7").add_site(
        name="wrist_camera",
        pos=[-0.06, 0, 0.025],
        quat=[2**-0.5, 0, -(2**-0.5), 0],
        size=[0.008] * 3,
    )
    frame = spec.worldbody.add_frame(name="arm_mount", pos=[-1, -4, 0.75])
    spec.attach(child, prefix="arm/", frame=frame)
    spec.worldbody.add_geom(
        name="arm_pedestal",
        type=mj.mjtGeom.mjGEOM_BOX,
        pos=[-1, -4, 0.375],
        size=[0.4, 0.4, 0.375],
        rgba=[0.3, 0.34, 0.35, 1],
    )


class ArticulatedRobot:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.root = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "arm/link_base")
        self.joints = [
            i
            for i in range(model.njnt)
            if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) or "").startswith("arm/")
        ]
        self.actuators = [
            i
            for i in range(model.nu)
            if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) or "").startswith(
                "arm/"
            )
        ]
        key = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "arm/home")
        if self.root < 0 or key < 0 or not self.actuators:
            raise ValueError("xArm7 requires its root, home keyframe and actuators")
        for joint in self.joints:
            adr = model.jnt_qposadr[joint]
            data.qpos[adr] = model.key_qpos[key, adr]
        self.target = model.key_ctrl[key, self.actuators].copy()
        self.apply()

    def command(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != self.target.shape or not np.isfinite(values).all():
            raise ValueError("Expected one finite control value per arm actuator")
        limits = self.model.actuator_ctrlrange[self.actuators]
        if np.any(values < limits[:, 0]) or np.any(values > limits[:, 1]):
            raise ValueError("Arm command exceeds actuator control limits")
        self.target = values.copy()

    def apply(self):
        self.data.ctrl[self.actuators] = self.target

    def state(self):
        m, d = self.model, self.data
        return dict(
            robot_id="arm",
            controller="MuJoCo joint servo · operator setpoints",
            names=[mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, i) for i in self.joints],
            position_rad=d.qpos[m.jnt_qposadr[self.joints]].tolist(),
            velocity_rad_s=d.qvel[m.jnt_dofadr[self.joints]].tolist(),
            effort_nm=d.qfrc_actuator[m.jnt_dofadr[self.joints]].tolist(),
            actuator_names=[
                mj.mj_id2name(m, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in self.actuators
            ],
            control=self.target.tolist(),
            control_limits=m.actuator_ctrlrange[self.actuators].tolist(),
            actuator_force=d.actuator_force[self.actuators].tolist(),
        )
