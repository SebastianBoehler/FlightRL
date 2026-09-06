"""Versioned raw-observation and action-head contracts for policy families."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, prod
import re

from flightrl.artifact_identity import bind_payload


POLICY_IO_SCHEMA = "flightrl.policy_io.v1"
_NAME = re.compile(r"[a-z][a-z0-9_]*")


class SignalRole(str, Enum):
    SENSOR = "sensor"
    VEHICLE = "vehicle"
    EMBODIMENT = "embodiment"
    GOAL = "goal"
    NEIGHBOR = "neighbor"


class SignalDType(str, Enum):
    UINT8 = "uint8"
    INT16 = "int16"
    INT32 = "int32"
    FLOAT32 = "float32"

    @property
    def bytes_per_element(self) -> int:
        return {
            SignalDType.UINT8: 1,
            SignalDType.INT16: 2,
            SignalDType.INT32: 4,
            SignalDType.FLOAT32: 4,
        }[self]


class SignalEncoding(str, Enum):
    RAW = "raw"
    CALIBRATED = "calibrated"
    DISCRETE = "discrete"
    DERIVED = "derived"


class ActionMode(str, Enum):
    VELOCITY_YAW_RATE = "velocity_yaw_rate"
    BODY_RATES_THRUST = "body_rates_thrust"
    DIRECT_MOTOR_THRUST = "direct_motor_thrust"


@dataclass(frozen=True, slots=True)
class InputSignal:
    name: str
    role: SignalRole
    dtype: SignalDType
    shape: tuple[int, ...]
    unit: str
    frame: str
    sample_rate_hz: float
    encoding: SignalEncoding
    normalization_scale: float = 1.0
    normalization_offset: float = 0.0

    def __post_init__(self) -> None:
        if _NAME.fullmatch(self.name) is None:
            raise ValueError("signal name must be lowercase snake_case")
        if not isinstance(self.role, SignalRole):
            raise TypeError("signal role must be a SignalRole")
        if not isinstance(self.dtype, SignalDType):
            raise TypeError("signal dtype must be a SignalDType")
        if not isinstance(self.encoding, SignalEncoding):
            raise TypeError("signal encoding must be a SignalEncoding")
        if self.encoding is SignalEncoding.DERIVED:
            raise ValueError(
                "policy inputs permit only identity or affine calibration, not derived features"
            )
        if not 1 <= len(self.shape) <= 3 or any(
            type(size) is not int or size <= 0 for size in self.shape
        ):
            raise ValueError("signal shape must contain one to three positive dimensions")
        if not self.unit or not self.frame:
            raise ValueError("signal unit and frame must be explicit")
        if not isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError("signal sample rate must be finite and positive")
        if not isfinite(self.normalization_scale) or self.normalization_scale == 0.0:
            raise ValueError("signal normalization scale must be finite and nonzero")
        if not isfinite(self.normalization_offset):
            raise ValueError("signal normalization offset must be finite")

    @property
    def element_count(self) -> int:
        return prod(self.shape)

    @property
    def byte_count(self) -> int:
        return self.element_count * self.dtype.bytes_per_element


@dataclass(frozen=True, slots=True)
class ActionSpec:
    mode: ActionMode
    fields: tuple[str, ...]
    rate_hz: float
    normalized_bounds: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ActionMode):
            raise TypeError("action mode must be an ActionMode")
        if not self.fields or any(_NAME.fullmatch(field) is None for field in self.fields):
            raise ValueError("action fields must be lowercase snake_case")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("action fields must be unique")
        if not isfinite(self.rate_hz) or self.rate_hz <= 0.0:
            raise ValueError("action rate must be finite and positive")
        low, high = self.normalized_bounds
        if not isfinite(low) or not isfinite(high) or low >= high:
            raise ValueError("action bounds must be finite and ordered")
        expected_width = {
            ActionMode.VELOCITY_YAW_RATE: 4,
            ActionMode.BODY_RATES_THRUST: 4,
        }.get(self.mode)
        if expected_width is not None and len(self.fields) != expected_width:
            raise ValueError(f"{self.mode.value} requires {expected_width} outputs")

    @classmethod
    def direct_motor_thrust(cls, *, rotor_count: int, rate_hz: float) -> ActionSpec:
        if type(rotor_count) is not int or rotor_count <= 0:
            raise ValueError("direct motor control requires a positive rotor count")
        return cls(
            mode=ActionMode.DIRECT_MOTOR_THRUST,
            fields=tuple(f"motor_{index}" for index in range(rotor_count)),
            rate_hz=rate_hz,
            normalized_bounds=(0.0, 1.0),
        )

    @classmethod
    def body_rates_and_thrust(cls, *, rate_hz: float) -> ActionSpec:
        return cls(
            mode=ActionMode.BODY_RATES_THRUST,
            fields=("roll_rate", "pitch_rate", "yaw_rate", "collective_thrust"),
            rate_hz=rate_hz,
            normalized_bounds=(-1.0, 1.0),
        )

    @classmethod
    def velocity_and_yaw_rate(cls, *, rate_hz: float) -> ActionSpec:
        return cls(
            mode=ActionMode.VELOCITY_YAW_RATE,
            fields=("vx", "vy", "vz", "yaw_rate"),
            rate_hz=rate_hz,
            normalized_bounds=(-1.0, 1.0),
        )


@dataclass(frozen=True, slots=True)
class PolicyIOContract:
    inputs: tuple[InputSignal, ...]
    action: ActionSpec

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("policy IO contract requires at least one input")
        if not all(isinstance(signal, InputSignal) for signal in self.inputs):
            raise TypeError("policy inputs must be InputSignal values")
        names = tuple(signal.name for signal in self.inputs)
        if len(set(names)) != len(names):
            raise ValueError("policy input names must be unique")
        if not isinstance(self.action, ActionSpec):
            raise TypeError("policy action must be an ActionSpec")


def compile_policy_io_contract(contract: PolicyIOContract) -> dict[str, object]:
    if not isinstance(contract, PolicyIOContract):
        raise TypeError("contract must be a PolicyIOContract")
    signals: list[dict[str, object]] = []
    offset = 0
    for signal in contract.inputs:
        signals.append(
            {
                "name": signal.name,
                "role": signal.role.value,
                "wire_dtype": signal.dtype.value,
                "shape": list(signal.shape),
                "element_count": signal.element_count,
                "byte_offset": offset,
                "byte_count": signal.byte_count,
                "unit": signal.unit,
                "frame": signal.frame,
                "sample_rate_hz": signal.sample_rate_hz,
                "encoding": signal.encoding.value,
                "model_mapping": {
                    "kind": "identity_or_affine",
                    "scale": signal.normalization_scale,
                    "offset": signal.normalization_offset,
                },
            }
        )
        offset += signal.byte_count
    return bind_payload(
        {
            "schema": POLICY_IO_SCHEMA,
            "authority": "simulation_and_training_contract_only",
            "deployment_authority": False,
            "observation": {
                "packing": "contiguous_no_padding",
                "bytes": offset,
                "feature_engineering": "none",
                "allowed_mapping": "identity_or_per_signal_affine_calibration",
                "temporal_metadata": {
                    "capture_time": "required_monotonic_uint64_us",
                    "sequence": "required_uint32",
                    "validity": "required_explicit_mask",
                    "missing_data": "never_zero_fill_as_valid",
                },
                "signals": signals,
            },
            "action": {
                "mode": contract.action.mode.value,
                "fields": list(contract.action.fields),
                "width": len(contract.action.fields),
                "rate_hz": contract.action.rate_hz,
                "normalized_bounds": list(contract.action.normalized_bounds),
                "consumer": "vehicle_specific_safety_and_actuator_adapter",
                "field_binding": "embodiment_actuator_order",
                "proposal_metadata": [
                    "sequence",
                    "generated_time_us",
                    "valid_until_us",
                    "uncertainty",
                    "health",
                ],
                "applied_action_feedback_required": True,
                "direct_motor_hardware_requires_separate_promotion": (
                    contract.action.mode is ActionMode.DIRECT_MOTOR_THRUST
                ),
            },
        }
    )
