from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flightrl.navigation.semantic_scene import SemanticScene


CRAZYFLIE_MJCF = """
<mujoco model="flightrl_crazyflie">
  <compiler angle="radian" inertiafromgeom="false"/>
  <option timestep="0.01" gravity="0 0 -9.81" integrator="RK4"/>
  <default>
    <geom contype="1" conaffinity="1" condim="3"/>
  </default>
  <asset>
    <material name="body" rgba="0.08 0.10 0.12 1"/>
    <material name="arm" rgba="0.22 0.25 0.28 1"/>
    <material name="rotor" rgba="0.05 0.05 0.05 1"/>
    <material name="floor" rgba="0.7 0.72 0.74 1"/>
    <material name="wall_x" rgba="0.82 0.84 0.86 1"/>
    <material name="wall_y" rgba="0.68 0.72 0.75 1"/>
    <material name="marker_dark" rgba="0.12 0.18 0.22 1"/>
    <material name="marker_light" rgba="0.78 0.25 0.12 1"/>
  </asset>
  <worldbody>
    <light name="top" pos="0 0 3"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="floor"/>
    <geom name="ceiling" type="box" pos="0 0 2.52" size="2 2 0.02" material="wall_x"/>
    <geom name="wall_x_neg" type="box" pos="-2.02 0 1.25" size="0.02 2 1.25" material="wall_x"/>
    <geom name="wall_x_pos" type="box" pos="2.02 0 1.25" size="0.02 2 1.25" material="wall_x"/>
    <geom name="wall_y_neg" type="box" pos="0 -2.02 1.25" size="2 0.02 1.25" material="wall_y"/>
    <geom name="wall_y_pos" type="box" pos="0 2.02 1.25" size="2 0.02 1.25" material="wall_y"/>
    <geom name="marker_x" type="box" pos="1.995 0.7 1.15" size="0.005 0.35 0.45"
          material="marker_dark" contype="0" conaffinity="0"/>
    <geom name="marker_y" type="box" pos="-0.7 1.995 0.8" size="0.4 0.005 0.25"
          material="marker_light" contype="0" conaffinity="0"/>
    <body name="crazyflie" pos="0 0 1">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="0.036" diaginertia="1.43e-5 1.43e-5 2.60e-5"/>
      <geom name="body" type="box" size="0.024 0.024 0.008" material="body"/>
      <geom name="arm_x" type="box" size="0.095 0.004 0.003" material="arm"/>
      <geom name="arm_y" type="box" size="0.004 0.095 0.003" material="arm"/>
      <geom name="rotor_front" type="cylinder" pos="0.09 0 0.01" size="0.024 0.002" material="rotor"/>
      <geom name="rotor_back" type="cylinder" pos="-0.09 0 0.01" size="0.024 0.002" material="rotor"/>
      <geom name="rotor_left" type="cylinder" pos="0 0.09 0.01" size="0.024 0.002" material="rotor"/>
      <geom name="rotor_right" type="cylinder" pos="0 -0.09 0.01" size="0.024 0.002" material="rotor"/>
      <camera name="aideck" pos="0.035 0 0.012" xyaxes="0 -1 0 0 0 1" fovy="63"/>
    </body>
  </worldbody>
</mujoco>
"""


def build_crazyflie_mjcf(scene: SemanticScene | None = None) -> str:
    if scene is None:
        return CRAZYFLIE_MJCF
    from .semantic_scene import add_semantic_scene_to_mjcf

    return add_semantic_scene_to_mjcf(CRAZYFLIE_MJCF, scene)


def load_crazyflie_model(timestep: float, scene: SemanticScene | None = None):
    mujoco = require_mujoco()
    model = mujoco.MjModel.from_xml_string(build_crazyflie_mjcf(scene))
    model.opt.timestep = float(timestep)
    return model


def require_mujoco():
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MuJoCo support requires the optional dependency: python -m pip install -e '.[mujoco]' --no-build-isolation"
        ) from exc
    return mujoco
