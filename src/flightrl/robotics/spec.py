"""Read mechanical identity from the actual compiled MJCF, never from display meshes."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import mujoco as mj


@dataclass(frozen=True)
class LinkSpec:
    name: str
    parent: str
    mass_kg: float
    center_of_mass_m: tuple
    principal_inertia_kg_m2: tuple
    inertia_rotation_wxyz: tuple


@dataclass(frozen=True)
class RobotSpec:
    name: str
    source_sha256: str
    links: tuple[LinkSpec, ...]
    joints: tuple[dict, ...]
    actuators: tuple[dict, ...]
    sensors: tuple[dict, ...]
    camera_mounts: tuple[dict, ...]

    def record(self):
        return asdict(self)


def import_robot(model, source: Path, name: str):
    root = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
    if root < 1:
        raise ValueError(f"Missing robot root: {name}")
    bodies = {root}
    for i in range(root + 1, model.nbody):
        if int(model.body_parentid[i]) in bodies:
            bodies.add(i)

    def label(kind, i):
        return mj.mj_id2name(model, kind, i) or ""

    links = []
    for i in sorted(bodies):
        if model.body_mass[i] <= 0 or (model.body_inertia[i] <= 0).any():
            raise ValueError("Positive masses and principal inertias required")
        links.append(
            LinkSpec(
                label(mj.mjtObj.mjOBJ_BODY, i),
                label(mj.mjtObj.mjOBJ_BODY, int(model.body_parentid[i])),
                float(model.body_mass[i]),
                tuple(model.body_ipos[i]),
                tuple(model.body_inertia[i]),
                tuple(model.body_iquat[i]),
            )
        )
    joint_ids = [i for i in range(model.njnt) if int(model.jnt_bodyid[i]) in bodies]
    joints = tuple(
        dict(
            name=label(mj.mjtObj.mjOBJ_JOINT, i),
            type=int(model.jnt_type[i]),
            axis=model.jnt_axis[i].tolist(),
            position_m=model.jnt_pos[i].tolist(),
            limited=bool(model.jnt_limited[i]),
            range=model.jnt_range[i].tolist(),
        )
        for i in joint_ids
    )

    def actuator_belongs(i):
        kind = int(model.actuator_trntype[i])
        target = int(model.actuator_trnid[i, 0])
        if kind == int(mj.mjtTrn.mjTRN_JOINT):
            return target in joint_ids
        if kind == int(mj.mjtTrn.mjTRN_TENDON):
            start, count = model.tendon_adr[target], model.tendon_num[target]
            wraps = range(start, start + count)
            owners = []
            for k in wraps:
                if model.wrap_type[k] == mj.mjtWrap.mjWRAP_JOINT:
                    owners.append(int(model.wrap_objid[k]) in joint_ids)
                else:
                    raise ValueError(
                        "Only joint and fixed-joint tendon transmissions are supported"
                    )
            if any(owners) and not all(owners):
                raise ValueError("Tendon spans multiple robot instances")
            return bool(owners) and all(owners)
        raise ValueError(f"Unsupported actuator transmission: {kind}")

    actuators = tuple(
        dict(
            name=label(mj.mjtObj.mjOBJ_ACTUATOR, i),
            transmission=int(model.actuator_trntype[i]),
            target=label(
                mj.mjtObj.mjOBJ_TENDON
                if model.actuator_trntype[i] == mj.mjtTrn.mjTRN_TENDON
                else mj.mjtObj.mjOBJ_JOINT,
                int(model.actuator_trnid[i, 0]),
            ),
            gear=model.actuator_gear[i].tolist(),
            dynamics_type=int(model.actuator_dyntype[i]),
            dynamics=model.actuator_dynprm[i].tolist(),
            control_range=model.actuator_ctrlrange[i].tolist(),
            control_limited=bool(model.actuator_ctrllimited[i]),
            force_limited=bool(model.actuator_forcelimited[i]),
            gain=model.actuator_gainprm[i].tolist(),
            bias=model.actuator_biasprm[i].tolist(),
            force_range=model.actuator_forcerange[i].tolist(),
        )
        for i in range(model.nu)
        if actuator_belongs(i)
    )
    sensors = tuple(
        dict(
            name=label(mj.mjtObj.mjOBJ_SENSOR, i),
            dimension=int(model.sensor_dim[i]),
            type=int(model.sensor_type[i]),
        )
        for i in range(model.nsensor)
        if label(mj.mjtObj.mjOBJ_SENSOR, i).startswith(name + "_")
        or (
            name == "rover"
            and label(mj.mjtObj.mjOBJ_SENSOR, i)
            in ("left_encoder", "right_encoder", "left_speed", "right_speed")
        )
    )
    return RobotSpec(
        name,
        hashlib.sha256(source.read_bytes()).hexdigest(),
        tuple(links),
        joints,
        actuators,
        sensors,
        tuple(
            dict(
                name=label(mj.mjtObj.mjOBJ_SITE, i),
                body=label(mj.mjtObj.mjOBJ_BODY, int(model.site_bodyid[i])),
                position_m=model.site_pos[i].tolist(),
                rotation_wxyz=model.site_quat[i].tolist(),
            )
            for i in range(model.nsite)
            if int(model.site_bodyid[i]) in bodies
            and label(mj.mjtObj.mjOBJ_SITE, i).endswith("_camera")
        ),
    )
