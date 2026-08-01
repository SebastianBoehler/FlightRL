from __future__ import annotations

from dataclasses import dataclass


ROOM_PROFILES = ("standard", "diverse")


@dataclass(frozen=True, slots=True)
class SemanticRoomGenerationConfig:
    width_range_m: tuple[float, float] = (3.8, 5.4)
    depth_range_m: tuple[float, float] = (3.8, 5.4)
    height_range_m: tuple[float, float] = (2.35, 2.85)
    flight_altitude_range_m: tuple[float, float] = (0.75, 1.25)
    approach_clearance_m: float = 0.65
    boundary_margin_m: float = 0.16
    obstacle_count_range: tuple[int, int] = (0, 0)
    obstacle_profiles: tuple[str, ...] = ("crate",)
    target_scale_range: tuple[float, float] = (1.0, 1.0)
    monitor_center_height_range_m: tuple[float, float] = (1.13, 1.13)
    appearance_contrast_range: tuple[float, float] = (0.08, 0.18)
    checker_repeat_range: tuple[float, float] = (3.0, 7.0)

    def __post_init__(self) -> None:
        low, high = self.obstacle_count_range
        if low < 0 or high < low:
            raise ValueError("obstacle count range must be ordered and non-negative")
        unknown = set(self.obstacle_profiles) - {
            "crate",
            "cabinet",
            "partition",
            "column",
        }
        if not self.obstacle_profiles or unknown:
            raise ValueError(f"unknown obstacle profiles: {sorted(unknown)}")
        for name, values in (
            ("target scale", self.target_scale_range),
            ("appearance contrast", self.appearance_contrast_range),
            ("checker repeat", self.checker_repeat_range),
        ):
            if values[0] <= 0.0 or values[1] < values[0]:
                raise ValueError(f"{name} range must be ordered and positive")

    @classmethod
    def for_profile(cls, profile: str) -> SemanticRoomGenerationConfig:
        if profile == "standard":
            return cls(obstacle_count_range=(2, 4))
        if profile == "diverse":
            return cls(
                width_range_m=(3.2, 6.8),
                depth_range_m=(3.2, 6.8),
                height_range_m=(2.25, 3.20),
                flight_altitude_range_m=(0.65, 1.35),
                obstacle_count_range=(1, 6),
                obstacle_profiles=("crate", "cabinet", "partition", "column"),
                target_scale_range=(0.70, 1.35),
                monitor_center_height_range_m=(0.85, 1.75),
                appearance_contrast_range=(0.04, 0.32),
                checker_repeat_range=(1.5, 10.0),
            )
        raise ValueError(f"room profile must be one of {ROOM_PROFILES}, got {profile!r}")
