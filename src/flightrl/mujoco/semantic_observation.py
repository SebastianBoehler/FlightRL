from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flightrl.mujoco.camera_model import randomize_gray4_frame
from flightrl.navigation.room_generation import SEMANTIC_TARGET_CATEGORIES
from flightrl.navigation.spatial_memory import SpatialMemoryConfig
from flightrl.sixdof.env import quat_to_yaw
from flightrl.sixdof.geometry import quat_to_matrix
from flightrl.vision import VisionObservationConfig


PROPRIOCEPTION_DIM = 13
COMMAND_DIM = len(SEMANTIC_TARGET_CATEGORIES)
GROUNDING_CONFIDENCE_INDEX = 11
GROUNDING_HORIZONTAL_ERROR_INDEX = 12


@dataclass(frozen=True, slots=True)
class SemanticStudentObservationLayout:
    vision: VisionObservationConfig
    spatial_memory: SpatialMemoryConfig

    @property
    def vision_slice(self) -> slice:
        return slice(0, self.vision.flat_dim)

    @property
    def map_slice(self) -> slice:
        start = self.vision.flat_dim
        return slice(start, start + self.spatial_memory.flat_dim)

    @property
    def proprioception_slice(self) -> slice:
        start = self.map_slice.stop
        return slice(start, start + PROPRIOCEPTION_DIM)

    @property
    def command_slice(self) -> slice:
        start = self.proprioception_slice.stop
        return slice(start, start + COMMAND_DIM)

    @property
    def flat_dim(self) -> int:
        return self.command_slice.stop


def proprioception(sim) -> np.ndarray:
    rotation = quat_to_matrix(sim.quaternion)
    body_velocity = np.einsum("nji,nj->ni", rotation, sim.velocity, optimize=True)
    yaw = quat_to_yaw(sim.quaternion)
    room_height = max(sim.room.z_max - sim.room.z_min, 1e-6)
    values = np.column_stack(
        (
            np.clip(body_velocity / 2.0, -1.0, 1.0),
            np.clip(sim.body_rates / sim.max_rate, -1.0, 1.0),
            np.clip((sim.position[:, 2] - sim.room.z_min) / room_height, 0.0, 1.0),
            np.sin(yaw),
            np.cos(yaw),
            sim.previous_action[:, :2],
            np.zeros((sim.num_envs, 2), dtype=np.float32),
        )
    )
    if values.shape[1] != PROPRIOCEPTION_DIM:
        raise RuntimeError(
            f"semantic proprioception has {values.shape[1]} values, expected {PROPRIOCEPTION_DIM}"
        )
    return values.astype(np.float32)


def write_semantic_observations(
    *,
    observations: np.ndarray,
    layout: SemanticStudentObservationLayout,
    renderer,
    sim,
    encoders,
    memories,
    odometry,
    target_category_indices: np.ndarray,
    target_means: np.ndarray,
    gammas: np.ndarray,
    rng: np.random.Generator,
) -> None:
    state = proprioception(sim)
    commands = np.eye(
        len(SEMANTIC_TARGET_CATEGORIES),
        dtype=np.float32,
    )[target_category_indices]
    for index, encoder in enumerate(encoders):
        renderer.update_scene(sim.data[index], camera="aideck")
        rgb = renderer.render().astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        frame = randomize_gray4_frame(
            gray,
            target_mean=float(target_means[index]),
            gamma=float(gammas[index]),
            rng=rng,
        )
        observations[index, layout.vision_slice] = encoder.encode_flat(frame)
        observations[index, layout.map_slice] = (
            memories[index]
            .local_map(
                odometry.position_xy[index],
                float(odometry.yaw[index]),
            )
            .reshape(-1)
        )
        observations[index, layout.proprioception_slice] = state[index]
        observations[index, layout.command_slice] = commands[index]
