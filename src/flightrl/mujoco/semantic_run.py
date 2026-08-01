from __future__ import annotations

from dataclasses import dataclass

from flightrl.mujoco.semantic_evaluation import (
    evaluate_semantic_policy,
)
from flightrl.mujoco.semantic_puffer_driver import SemanticPufferDriver
from flightrl.mujoco.semantic_teacher_evaluation import evaluate_semantic_teacher


@dataclass(frozen=True, slots=True)
class SemanticDriverConfig:
    seed: int
    room_count: int
    agents_per_room: int
    offset: int = 0
    active_exploration: bool = False
    vision_width: int = 64
    vision_height: int = 48
    room_profile: str = "standard"

    def build(self) -> SemanticPufferDriver:
        room_seeds = tuple(
            self.seed + self.offset + index for index in range(self.room_count)
        )
        return SemanticPufferDriver(
            room_seeds=room_seeds,
            agents_per_room=self.agents_per_room,
            seed=self.seed + self.offset,
            active_exploration=self.active_exploration,
            vision_width=self.vision_width,
            vision_height=self.vision_height,
            room_profile=self.room_profile,
        )


def evaluate_policy_rooms(
    policy,
    config: SemanticDriverConfig,
    *,
    steps: int,
    mode: str,
) -> dict[str, float]:
    driver = config.build()
    try:
        return evaluate_semantic_policy(
            policy,
            driver,
            steps=steps,
            mode=mode,
        )
    finally:
        driver.close()


def evaluate_teacher_rooms(
    config: SemanticDriverConfig,
    *,
    steps: int,
) -> dict[str, float]:
    driver = config.build()
    try:
        return evaluate_semantic_teacher(driver, steps=steps)
    finally:
        driver.close()
