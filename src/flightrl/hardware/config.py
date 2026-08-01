from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

import tomllib

from .errors import HardwareConfigError


DEFAULT_LOG_VARIABLES = (
    "stabilizer.roll",
    "stabilizer.pitch",
    "stabilizer.yaw",
    "stabilizer.thrust",
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stateEstimate.vx",
    "stateEstimate.vy",
    "stateEstimate.vz",
    "stateEstimate.roll",
    "stateEstimate.pitch",
    "stateEstimate.yaw",
    "stateEstimate.qx",
    "stateEstimate.qy",
    "stateEstimate.qz",
    "stateEstimate.qw",
    "acc.x",
    "acc.y",
    "acc.z",
    "gyro.x",
    "gyro.y",
    "gyro.z",
    "pm.vbat",
    "pm.vbatMV",
    "pm.batteryLevel",
    "pm.state",
    "pm.chargeCurrent",
    "pm.extVbat",
    "pm.extVbatMV",
    "pm.extCurr",
    "radio.rssi",
    "radio.isConnected",
    "radio.numRxUc",
    "radio.numRxBc",
    "supervisor.info",
    "sys.canfly",
    "sys.isFlying",
    "sys.isTumbled",
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
    "range.zrange",
    "controller.cmd_thrust",
    "controller.cmd_roll",
    "controller.cmd_pitch",
    "controller.cmd_yaw",
    "controller.actuatorThrust",
    "ctrltarget.x",
    "ctrltarget.y",
    "ctrltarget.z",
    "ctrltarget.vx",
    "ctrltarget.vy",
    "ctrltarget.vz",
    "ctrltarget.roll",
    "ctrltarget.pitch",
    "ctrltarget.yaw",
    "ctrltarget.thrust",
    "motor.m1",
    "motor.m2",
    "motor.m3",
    "motor.m4",
    "motor.m1req",
    "motor.m2req",
    "motor.m3req",
    "motor.m4req",
    "rpm.m1",
    "rpm.m2",
    "rpm.m3",
    "rpm.m4",
    "health.motorPass",
    "health.batteryPass",
    "health.batterySag",
    "health.motorTestCount",
    "kalman.varX",
    "kalman.varY",
    "kalman.varZ",
    "kalman.rtFinal",
)

DEFAULT_LOG_VARIABLE_TYPES = {
    "pm.vbatMV": "uint16_t",
    "pm.batteryLevel": "uint8_t",
    "pm.state": "int8_t",
    "pm.extVbatMV": "uint16_t",
    "radio.rssi": "uint8_t",
    "radio.isConnected": "uint8_t",
    "radio.numRxUc": "uint16_t",
    "radio.numRxBc": "uint16_t",
    "supervisor.info": "uint16_t",
    "sys.canfly": "uint8_t",
    "sys.isFlying": "uint8_t",
    "sys.isTumbled": "uint8_t",
    "motor.m1": "uint16_t",
    "motor.m2": "uint16_t",
    "motor.m3": "uint16_t",
    "motor.m4": "uint16_t",
    "motor.m1req": "int32_t",
    "motor.m2req": "int32_t",
    "motor.m3req": "int32_t",
    "motor.m4req": "int32_t",
    "rpm.m1": "uint16_t",
    "rpm.m2": "uint16_t",
    "rpm.m3": "uint16_t",
    "rpm.m4": "uint16_t",
    "health.motorPass": "uint8_t",
    "health.batteryPass": "uint8_t",
    "health.motorTestCount": "uint16_t",
}


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
    expect_ai_deck: bool = False
    expect_zranger: bool = False


@dataclass(slots=True)
class CrazyflieLoggingConfig:
    period_ms: int = 50
    output_dir: str = "artifacts/crazyflie_logs"
    variables: tuple[str, ...] = DEFAULT_LOG_VARIABLES
    variable_types: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LOG_VARIABLE_TYPES))


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
    if not isinstance(uri, str) or not (uri.startswith("radio://") or uri.startswith("usb://")):
        raise HardwareConfigError("radio.uri must start with radio:// or usb://")
    if not isinstance(config.radio.cache_dir, str) or not config.radio.cache_dir:
        raise HardwareConfigError("radio.cache_dir must not be empty")

    safety = config.safety
    _range("safety.default_height_m", safety.default_height_m, low=0.1, high=0.8)
    _range("safety.velocity_m_s", safety.velocity_m_s, low=0.05, high=0.5)
    _range("safety.turn_rate_deg_s", safety.turn_rate_deg_s, low=5.0, high=120.0)
    _range("safety.turn_angle_deg", safety.turn_angle_deg, low=1.0, high=90.0)
    _range("safety.hover_s", safety.hover_s, low=0.0, high=10.0)
    _range("safety.max_flight_s", safety.max_flight_s, low=1.0, high=60.0)
    if type(safety.requires_manual_confirm) is not bool:
        raise HardwareConfigError("safety.requires_manual_confirm must be a boolean")

    for name in (
        "expect_flow_deck",
        "expect_multiranger",
        "expect_ai_deck",
        "expect_zranger",
    ):
        if type(getattr(config.decks, name)) is not bool:
            raise HardwareConfigError(f"decks.{name} must be a boolean")

    if type(config.logging.period_ms) is not int or not 10 <= config.logging.period_ms <= 1000:
        raise HardwareConfigError("logging.period_ms must be between 10 and 1000")
    if not isinstance(config.logging.output_dir, str) or not config.logging.output_dir:
        raise HardwareConfigError("logging.output_dir must not be empty")
    if not config.logging.variables or not all(
        isinstance(value, str) and value and "." in value
        for value in config.logging.variables
    ):
        raise HardwareConfigError(
            "logging.variables must contain nonempty group.name strings"
        )


def _load_logging(raw: dict[str, Any]) -> CrazyflieLoggingConfig:
    data = dict(raw)
    if "variables" in data:
        variables = data["variables"]
        if not isinstance(variables, list) or not all(
            isinstance(value, str) for value in variables
        ):
            raise HardwareConfigError("logging.variables must be a list of strings")
        data["variables"] = tuple(variables)
    variable_types = dict(DEFAULT_LOG_VARIABLE_TYPES)
    if "variable_types" in data:
        overrides = data["variable_types"]
        if not isinstance(overrides, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in overrides.items()
        ):
            raise HardwareConfigError(
                "logging.variable_types must map strings to strings"
            )
        variable_types.update(overrides)
    data["variable_types"] = variable_types
    return CrazyflieLoggingConfig(**data)


def _range(name: str, value: float, *, low: float, high: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise HardwareConfigError(f"{name} must be a finite number")
    if value < low or value > high:
        raise HardwareConfigError(f"{name} must be between {low} and {high}")
