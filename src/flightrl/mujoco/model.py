from __future__ import annotations


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
  </asset>
  <worldbody>
    <light name="top" pos="0 0 3"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="floor"/>
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
    </body>
  </worldbody>
</mujoco>
"""


def load_crazyflie_model(timestep: float):
    mujoco = require_mujoco()
    model = mujoco.MjModel.from_xml_string(CRAZYFLIE_MJCF)
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
