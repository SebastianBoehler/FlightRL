from __future__ import annotations

from dataclasses import dataclass

import torch

from flightrl.puffer4_door_grounding_metrics import (
    fixed_door_grounder_gate,
    grounding_metrics,
)

MUJOCO_TRAIN_ROOM_SEEDS = tuple(range(20_001, 20_257))
MUJOCO_SELECTION_ROOM_SEEDS = tuple(range(30_001, 30_065))
MUJOCO_EVALUATION_ROOM_SEEDS = tuple(range(40_001, 40_129))
MUJOCO_SAMPLES_PER_ROOM = 16


@dataclass(frozen=True, slots=True)
class MujocoGroundingReplay:
    frames: torch.Tensor
    labels: torch.Tensor
    room_seeds: tuple[int, ...]

    @property
    def sample_count(self) -> int:
        return int(self.frames.shape[0])


def collect_mujoco_grounding_replay(
    *,
    room_seeds: tuple[int, ...],
    seed: int,
    samples_per_room: int = MUJOCO_SAMPLES_PER_ROOM,
) -> MujocoGroundingReplay:
    from flightrl.mujoco.door_observability import (
        collect_synthetic_door_dataset,
    )

    dataset = collect_synthetic_door_dataset(
        room_seeds=room_seeds,
        samples_per_room=samples_per_room,
        seed=seed,
    )
    return MujocoGroundingReplay(
        frames=torch.from_numpy(dataset.frames),
        labels=torch.from_numpy(dataset.labels),
        room_seeds=room_seeds,
    )


@torch.no_grad()
def evaluate_mujoco_grounder(
    grounder,
    replay: MujocoGroundingReplay,
    *,
    visibility_threshold: float,
) -> dict[str, float]:
    device = next(grounder.parameters()).device
    logits = grounder(replay.frames.to(device)).cpu()
    return grounding_metrics(
        logits,
        replay.labels,
        visibility_threshold=visibility_threshold,
    )


def collect_default_mujoco_replays(
) -> tuple[MujocoGroundingReplay, MujocoGroundingReplay]:
    return (
        collect_mujoco_grounding_replay(
            room_seeds=MUJOCO_TRAIN_ROOM_SEEDS,
            seed=20_101,
        ),
        collect_mujoco_grounding_replay(
            room_seeds=MUJOCO_SELECTION_ROOM_SEEDS,
            seed=30_101,
        ),
    )


def sample_mixed_grounding_batch(
    native_frames: torch.Tensor,
    native_labels: torch.Tensor,
    mujoco: MujocoGroundingReplay,
    *,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    native_count = batch_size // 2
    mujoco_count = batch_size - native_count
    native_indices = torch.randint(
        native_frames.shape[0],
        (native_count,),
        generator=generator,
    )
    mujoco_indices = torch.randint(
        mujoco.sample_count,
        (mujoco_count,),
        generator=generator,
    )
    return (
        torch.cat(
            (
                native_frames[native_indices].float() / 255.0,
                mujoco.frames[mujoco_indices],
            )
        ),
        torch.cat(
            (
                native_labels[native_indices],
                mujoco.labels[mujoco_indices],
            )
        ),
    )


def mixed_grounding_selection_score(
    native_metrics: dict[str, float],
    mujoco_metrics: dict[str, float],
) -> float:
    return min(
        native_metrics.get("visibility_auroc", 0.0) - 0.90,
        0.12 - native_metrics.get("centroid_median_error_widths", 1.0),
        mujoco_metrics.get("visibility_auroc", 0.0) - 0.90,
        0.12 - mujoco_metrics.get("centroid_median_error_widths", 1.0),
    )


def combined_grounder_gate(
    native_metrics: dict[str, float],
    mujoco_metrics: dict[str, float],
) -> dict:
    native = fixed_door_grounder_gate(native_metrics)
    mujoco = fixed_door_grounder_gate(mujoco_metrics)
    return {
        "passed": native["passed"] and mujoco["passed"],
        "native": native,
        "mujoco": mujoco,
        "failures": [
            *(f"native:{name}" for name in native["failures"]),
            *(f"mujoco:{name}" for name in mujoco["failures"]),
        ],
    }
