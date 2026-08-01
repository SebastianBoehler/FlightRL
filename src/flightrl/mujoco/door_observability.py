from __future__ import annotations

import gc
from dataclasses import dataclass
from math import atan2, pi

import numpy as np

from flightrl.mujoco.camera_model import randomize_gray4_frame
from flightrl.mujoco.env import MuJoCoCrazyflieEnv
from flightrl.navigation.room_config import SemanticRoomGenerationConfig
from flightrl.navigation.room_generation import generate_semantic_room
from flightrl.semantic.door_observability import labels_from_segmentation
from flightrl.sixdof.env import euler_to_quat


@dataclass(frozen=True, slots=True)
class SyntheticDoorDataset:
    frames: np.ndarray
    labels: np.ndarray
    room_seeds: np.ndarray


def collect_synthetic_door_dataset(
    *,
    room_seeds: tuple[int, ...],
    samples_per_room: int,
    seed: int,
    width: int = 64,
    height: int = 48,
) -> SyntheticDoorDataset:
    if not room_seeds:
        raise ValueError("room_seeds cannot be empty")
    if samples_per_room <= 1:
        raise ValueError("samples_per_room must be greater than one")
    rng = np.random.default_rng(seed)
    frames: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sample_room_seeds: list[int] = []
    config = SemanticRoomGenerationConfig.for_profile("diverse")
    for room_seed in room_seeds:
        scene = generate_semantic_room(int(room_seed), config)
        room_frames, room_labels = _collect_room(
            scene=scene,
            samples=samples_per_room,
            width=width,
            height=height,
            rng=rng,
        )
        frames.extend(room_frames)
        labels.extend(room_labels)
        sample_room_seeds.extend([int(room_seed)] * samples_per_room)
    return SyntheticDoorDataset(
        frames=np.stack(frames).astype(np.float32)[:, None, ...] / 255.0,
        labels=np.stack(labels).astype(np.float32),
        room_seeds=np.asarray(sample_room_seeds, dtype=np.int64),
    )


def _collect_room(
    *,
    scene,
    samples: int,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    env = MuJoCoCrazyflieEnv(num_envs=1, semantic_scene=scene, seed=0)
    mujoco = env.mujoco
    target_ids = tuple(
        geom_id
        for geom_id in range(env.model.ngeom)
        if (
            name := mujoco.mj_id2name(
                env.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_id,
            )
        )
        and name.startswith("semantic_door_0")
    )
    if not target_ids:
        raise RuntimeError("semantic MuJoCo scene is missing door geometry")
    _, door = scene.object_by_name("door")
    target = np.asarray(door.bounds.center, dtype=np.float64)
    frames: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with (
        mujoco.Renderer(env.model, height=height, width=width) as rgb_renderer,
        mujoco.Renderer(env.model, height=height, width=width) as seg_renderer,
    ):
        seg_renderer.enable_segmentation_rendering()
        for sample_index in range(samples):
            position = _sample_free_position(env, rng)
            yaw_to_target = atan2(
                target[1] - position[1],
                target[0] - position[0],
            )
            if sample_index % 2 == 0:
                yaw = yaw_to_target + rng.uniform(-0.42, 0.42)
            else:
                yaw = yaw_to_target + pi + rng.uniform(-0.65, 0.65)
            _set_pose(env, position, yaw, rng)
            rgb_renderer.update_scene(env.data[0], camera="aideck")
            rgb = rgb_renderer.render().astype(np.float32)
            gray = (
                0.299 * rgb[..., 0]
                + 0.587 * rgb[..., 1]
                + 0.114 * rgb[..., 2]
            )
            frame = randomize_gray4_frame(
                gray,
                target_mean=float(rng.uniform(35.0, 90.0)),
                gamma=float(rng.uniform(0.8, 1.2)),
                rng=rng,
            )
            seg_renderer.update_scene(env.data[0], camera="aideck")
            segmentation = seg_renderer.render()
            label = labels_from_segmentation(
                segmentation,
                target_geom_id=target_ids,
            )
            frames.append(frame)
            labels.append(label.as_array())
    del rgb_renderer
    del seg_renderer
    del env
    gc.collect()
    return frames, labels


def _sample_free_position(
    env: MuJoCoCrazyflieEnv,
    rng: np.random.Generator,
) -> np.ndarray:
    room = env.room
    for _ in range(128):
        position = np.asarray(
            (
                rng.uniform(room.x_min + 0.35, room.x_max - 0.35),
                rng.uniform(room.y_min + 0.35, room.y_max - 0.35),
                rng.uniform(
                    room.z_min + 0.55,
                    min(room.z_max - 0.35, 1.45),
                ),
            ),
            dtype=np.float64,
        )
        if bool(room.contains(position[None, :], margin=0.25)[0]):
            return position
    raise RuntimeError("could not sample a collision-free observability pose")


def _set_pose(
    env: MuJoCoCrazyflieEnv,
    position: np.ndarray,
    yaw: float,
    rng: np.random.Generator,
) -> None:
    roll = np.asarray((rng.uniform(-0.05, 0.05),), dtype=np.float32)
    pitch = np.asarray((rng.uniform(-0.05, 0.05),), dtype=np.float32)
    quaternion = euler_to_quat(
        roll,
        pitch,
        np.asarray((yaw,), dtype=np.float32),
    )[0]
    data = env.data[0]
    env.mujoco.mj_resetData(env.model, data)
    data.qpos[:3] = position
    data.qpos[3:7] = quaternion
    data.qvel[:] = 0.0
    env.mujoco.mj_forward(env.model, data)
