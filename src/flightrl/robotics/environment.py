"""Renderer-independent reset, physics-step and sensor-observation environment."""

from .world import RobotWorld
from .industrial import equipment
from .utility_site import utility_site
from .mission import InspectionMission
from .sensor_rig import SensorRig
from .sensing import body_sensors


class RobotEnvironment:
    def __init__(self, seed=0, policy=None, industry=False, arm=False):
        self.policy = policy
        self.industry = industry
        self.arm = arm
        self.reset(seed)

    def reset(self, seed):
        self.seed = seed
        if self.industry:
            items, self.targets, site = utility_site(seed)
        else:
            items, self.targets = equipment(seed)
            site = None
        self.world = RobotWorld(items, site, self.arm)
        self.last_delivery = None
        self.mission = InspectionMission(self.world, self.targets, self.policy)
        self.sensor_rig = SensorRig(self.world, seed) if self.industry else None
        self.mission.control_estimate = self.sensor_rig
        self.reward_events = 0
        return self.state()

    def description(self):
        return {
            **self.world.render_description(),
            "targets": self.targets,
            "site": self.world.site,
        }

    def step(self):
        self.mission.apply()
        for _ in range(10):
            self.world.step()
        if self.sensor_rig:
            self.sensor_rig.update(self.world, 0.02)
        return self.state()

    def observe(self, frames, captured_state):
        delivery = (
            (frames, captured_state)
            if self.sensor_rig is None
            else self.sensor_rig.deliver(
                frames, captured_state, self.world.ticks * self.world.dt
            )
        )
        self.last_delivery = delivery
        if delivery is not None:
            delivery[1]["decision_time_s"] = self.world.ticks * self.world.dt
            delivery[1]["decision_tick"] = self.world.ticks
            self.mission.observe(*delivery)
        events = self.mission.events[self.reward_events :]
        self.reward_events = len(self.mission.events)
        return dict(
            reward=sum(1 if e["verified"] else -1 for e in events),
            terminated=len(self.mission.events) == 3
            or self.world.robot_collision_steps > 0,
            truncated=self.world.ticks * self.world.dt
            >= (180 if self.industry else 120),
            info=self.mission.status(),
        )

    def state(self):
        time_s = self.world.ticks * self.world.dt
        camera = {
            **self.world.cameras(),
            "time_s": time_s,
            "sequence": self.world.ticks,
            "wind_m_s": self.world.wind.tolist(),
            "contacts": int(self.world.data.ncon),
            "mode": "inspection",
        }
        state = dict(
            schema_version=2,
            clock_id="simulation",
            capture_time_ns=round(self.world.ticks * self.world.dt * 1e9),
            time_s=time_s,
            sequence=self.world.ticks,
            bodies=self.world.bodies(),
            camera=camera,
            proprio=[
                body_sensors(self.world, b).tolist()
                for b in (self.world.drone, self.world.rover)
            ],
            qpos=self.world.data.qpos.tolist(),
            qvel=self.world.data.qvel.tolist(),
            mocap_pos=self.world.data.mocap_pos.tolist(),
            mocap_quat=self.world.data.mocap_quat.tolist(),
            camera_poses=[
                dict(
                    position_m=self.world.data.site_xpos[i].tolist(),
                    rotation=self.world.data.site_xmat[i].tolist(),
                )
                for i in self.world.camera_sites
            ],
            encoder=self.world.data.qpos[
                self.world.model.jnt_qposadr[
                    [
                        self.world.model.joint(n).id
                        for n in ("left_wheel_joint", "right_wheel_joint")
                    ]
                ]
            ].tolist(),
            imu=self.world.data.sensordata.tolist(),
            arm=self.world.arm.state() if self.world.arm else None,
        )
        if self.sensor_rig:
            state["proprio"] = self.sensor_rig.proprio.tolist()
            state["estimation"] = dict(
                positions=self.sensor_rig.position.tolist(),
                quaternions=self.sensor_rig.q.tolist(),
                variance=self.sensor_rig.variance.tolist(),
            )
        return state
