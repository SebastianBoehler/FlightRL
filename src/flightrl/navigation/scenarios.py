from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NavigationThresholds:
    min_completed_fraction: float
    min_survival_fraction: float
    max_position_error_m: float
    min_clearance_p01_m: float
    max_action_saturation_fraction: float
    preferred_clearance_m: float


@dataclass(frozen=True)
class NavigationScenario:
    name: str
    task: str
    reset_profile: str
    description: str
    thresholds: NavigationThresholds
    seed_count: int = 128
    drones_per_env: int = 1
    observation_source: str = "range_telemetry"
    action_interface: str = "firmware_setpoint"
    tags: tuple[str, ...] = ()
    required_metrics: tuple[str, ...] = (
        "mean_completed_fraction",
        "mean_survival_fraction",
        "mean_position_error_m",
        "clearance_p01_m",
        "action_saturation_fraction",
    )


DEFAULT_NAVIGATION_SCENARIOS: tuple[NavigationScenario, ...] = (
    NavigationScenario(
        name="target_approach",
        task="position_yaw",
        reset_profile="position_yaw_wide",
        description="Reach a local target from varied starts while preserving yaw control.",
        thresholds=NavigationThresholds(0.90, 0.95, 0.45, 0.10, 0.12, 0.30),
        tags=("target", "command_following"),
    ),
    NavigationScenario(
        name="obstacle_room",
        task="obstacle_avoidance",
        reset_profile="obstacle_close_live",
        description="Navigate near room boundaries using ranger-style clearance signals.",
        thresholds=NavigationThresholds(0.88, 0.95, 0.55, 0.12, 0.14, 0.35),
        tags=("avoidance", "range"),
    ),
    NavigationScenario(
        name="vertical_clearance",
        task="obstacle_avoidance",
        reset_profile="obstacle_vertical_live",
        description="Handle top and bottom clearance pressure without camera input.",
        thresholds=NavigationThresholds(0.84, 0.94, 0.65, 0.08, 0.16, 0.28),
        tags=("avoidance", "vertical", "range"),
    ),
    NavigationScenario(
        name="recovery",
        task="obstacle_avoidance",
        reset_profile="obstacle_close_live",
        description="Recover from blocked starts before resuming target-directed motion.",
        thresholds=NavigationThresholds(0.80, 0.92, 0.75, 0.10, 0.18, 0.30),
        tags=("recovery", "range"),
    ),
    NavigationScenario(
        name="hold_or_land",
        task="position_yaw",
        reset_profile="position_yaw_medium",
        description="Slow down near the goal and hold a safe final setpoint.",
        thresholds=NavigationThresholds(0.92, 0.97, 0.30, 0.10, 0.10, 0.30),
        tags=("hold", "setpoint"),
    ),
)


def scenario_by_name(name: str) -> NavigationScenario:
    for scenario in DEFAULT_NAVIGATION_SCENARIOS:
        if scenario.name == name:
            return scenario
    known = ", ".join(scenario.name for scenario in DEFAULT_NAVIGATION_SCENARIOS)
    raise KeyError(f"unknown navigation scenario {name!r}; known scenarios: {known}")
