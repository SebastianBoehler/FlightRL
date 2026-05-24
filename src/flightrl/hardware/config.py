from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib

from .errors import HardwareConfigError


DEFAULT_LOG_VARIABLES = (
    "stabilizer.roll",
    "stabilizer.pitch",
    "stabilizer.yaw",
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stateEstimate.vx",
    "stateEstimate.vy",
    "stateEstimate.vz",
    "acc.x",
    "acc.y",
    "acc.z",
    "gyro.x",
    "gyro.y",
    "gyro.z",
    "pm.vbat",
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
)


@dataclass(slots=True)
class CrazyflieRadioConfig:
    uri: str = "radio://0/80/2M/E7E7E7E7E7"
    cache_dir: str = "artifacts/cflib_cache"


@dataclass(slots=True)
class CrazyflieSafetyConfig:
    default_height_m: float = 0.3
    velocity_m_s: float = 0.15
    turn_rate_deg_s: float = 45.0
    turn_angle_deg: float = 20.0
    hover_s: float = 2.0
    max_flight_s: float = 20.0
    requires_manual_confirm: bool = True


@dataclass(slots=True)
class CrazyflieDeckConfig:
    expect_flow_deck: bool = True
    expect_multiranger: bool = True


@dataclass(slots=True)
class CrazyflieLoggingConfig:
    period_ms: int = 50
    output_dir: str = "artifacts/crazyflie_logs"
    variables: tuple[str, ...] = DEFAULT_LOG_VARIABLES
    variable_types: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class CrazyflieHardwareConfig:
    radio: CrazyflieRadioConfig = field(default_factory=CrazyflieRadioConfig)
    safety: CrazyflieSafetyConfig = field(default_factory=CrazyflieSafetyConfig)
    decks: CrazyflieDeckConfig = field(default_factory=CrazyflieDeckConfig)
    logging: CrazyflieLoggingConfig = field(default_factory=CrazyflieLoggingConfig)


def load_hardware_config(path: str | Path) -> CrazyflieHardwareConfig:
    raw = tomllib.loads(Path(path).read_text())
    config = CrazyflieHardwareConfig(
        radio=CrazyflieRadioConfig(**raw.get("radio", {})),
        safety=CrazyflieSafetyConfig(**raw.get("safety", {})),
        decks=CrazyflieDeckConfig(**raw.get("decks", {})),
        logging=_load_logging(raw.get("logging", {})),
    )
    validate_hardware_config(config)
    return config


def validate_hardware_config(config: CrazyflieHardwareConfig) -> None:
    uri = config.radio.uri
    if not (uri.startswith("radio://") or uri.startswith("usb://")):
        raise HardwareConfigError("radio.uri must start with radio:// or usb://")
    if not config.radio.cache_dir:
        raise HardwareConfigError("radio.cache_dir must not be empty")

    safety = config.safety
    _range("safety.default_height_m", safety.default_height_m, low=0.1, high=0.8)
    _range("safety.velocity_m_s", safety.velocity_m_s, low=0.05, high=0.5)
    _range("safety.turn_rate_deg_s", safety.turn_rate_deg_s, low=5.0, high=120.0)
    _range("safety.turn_angle_deg", safety.turn_angle_deg, low=1.0, high=90.0)
    _range("safety.hover_s", safety.hover_s, low=0.0, high=10.0)
    _range("safety.max_flight_s", safety.max_flight_s, low=1.0, high=60.0)

    if config.logging.period_ms < 10 or config.logging.period_ms > 1000:
        raise HardwareConfigError("logging.period_ms must be between 10 and 1000")
    if not config.logging.variables:
        raise HardwareConfigError("logging.variables must include at least one variable")


def _load_logging(raw: dict[str, Any]) -> CrazyflieLoggingConfig:
    data = dict(raw)
    if "variables" in data:
        data["variables"] = tuple(str(value) for value in data["variables"])
    if "variable_types" in data:
        data["variable_types"] = {str(key): str(value) for key, value in data["variable_types"].items()}
    return CrazyflieLoggingConfig(**data)


def _range(name: str, value: float, *, low: float, high: float) -> None:
    if value < low or value > high:
        raise HardwareConfigError(f"{name} must be between {low} and {high}")
