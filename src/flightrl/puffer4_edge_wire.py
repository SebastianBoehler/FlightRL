from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from flightrl.puffer4_edge_schema import EDGE_MISSION_TOKEN_COUNT


UINT32_MODULUS = 1 << 32
UINT32_HALF_RANGE = 1 << 31
EDGE_INPUT_PACKET_BYTES = 1_635
EDGE_OUTPUT_PACKET_BYTES = 39


@dataclass(frozen=True, slots=True)
class EdgeTimingProfile:
    nominal_period_us: int
    minimum_period_us: int
    maximum_period_us: int
    max_frame_telemetry_skew_us: int
    max_proposal_age_us: int
    measurement_frames: int
    measurement_sha256: str

    def __post_init__(self) -> None:
        integers = (
            self.nominal_period_us,
            self.minimum_period_us,
            self.maximum_period_us,
            self.max_frame_telemetry_skew_us,
            self.max_proposal_age_us,
            self.measurement_frames,
        )
        if any(type(value) is not int for value in integers):
            raise ValueError("edge timing fields must be exact integers")
        if not (
            0 < self.minimum_period_us
            <= self.nominal_period_us
            <= self.maximum_period_us
            < UINT32_HALF_RANGE
        ):
            raise ValueError("edge timing periods are inconsistent")
        if not 0 <= self.max_frame_telemetry_skew_us <= self.maximum_period_us:
            raise ValueError("edge frame/telemetry skew is outside the period")
        if not 0 < self.max_proposal_age_us < UINT32_HALF_RANGE:
            raise ValueError("edge proposal age must fit unsigned clock ordering")
        if self.measurement_frames < 1_000:
            raise ValueError("edge timing binding requires at least 1000 frames")
        if re.fullmatch(r"[0-9a-f]{64}", self.measurement_sha256) is None:
            raise ValueError("edge timing measurement SHA-256 is invalid")

    def payload(self) -> dict[str, Any]:
        return {
            "status": "measured_and_bound",
            "clock_domain": "stm32_monotonic_uint32_us",
            "nominal_period_us": self.nominal_period_us,
            "minimum_period_us": self.minimum_period_us,
            "maximum_period_us": self.maximum_period_us,
            "max_frame_telemetry_skew_us": self.max_frame_telemetry_skew_us,
            "max_proposal_age_us": self.max_proposal_age_us,
            "measurement_frames": self.measurement_frames,
            "measurement_sha256": self.measurement_sha256,
        }


@dataclass(frozen=True, slots=True)
class EdgeInputStamp:
    frame_sequence: int
    capture_time_us: int
    telemetry_time_us: int
    mission_epoch: int
    arming_epoch: int
    target_id: int

    def __post_init__(self) -> None:
        for name in (
            "frame_sequence",
            "capture_time_us",
            "telemetry_time_us",
            "mission_epoch",
            "arming_epoch",
        ):
            validate_u32(getattr(self, name), name)
        if (
            type(self.target_id) is not int
            or not 0 <= self.target_id < EDGE_MISSION_TOKEN_COUNT
        ):
            raise ValueError("edge target ID is outside the approved v3 vocabulary")


@dataclass(frozen=True, slots=True)
class EdgeProposalStamp:
    source_frame_sequence: int
    source_capture_time_us: int
    mission_epoch: int
    arming_epoch: int
    source_target_id: int
    proposal_sequence: int

    def __post_init__(self) -> None:
        for name in (
            "source_frame_sequence",
            "source_capture_time_us",
            "mission_epoch",
            "arming_epoch",
            "proposal_sequence",
        ):
            validate_u32(getattr(self, name), name)
        if (
            type(self.source_target_id) is not int
            or not 0 <= self.source_target_id < EDGE_MISSION_TOKEN_COUNT
        ):
            raise ValueError(
                "edge source target ID is outside the approved v3 vocabulary"
            )


def validate_u32(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value < UINT32_MODULUS:
        raise ValueError(f"{label} must be uint32")
    return value


def forward_u32_delta(new: object, old: object, label: str) -> int:
    current = validate_u32(new, label)
    previous = validate_u32(old, label)
    delta = (current - previous) & 0xFFFF_FFFF
    if not 0 < delta < UINT32_HALF_RANGE:
        raise ValueError(f"{label} is duplicate or reordered")
    return delta


def validate_edge_input_order(
    current: EdgeInputStamp,
    previous: EdgeInputStamp | None,
    timing: EdgeTimingProfile,
) -> bool:
    if _absolute_u32_distance(
        current.capture_time_us,
        current.telemetry_time_us,
    ) > timing.max_frame_telemetry_skew_us:
        raise ValueError("edge frame and telemetry timestamps are misaligned")
    if previous is None:
        return True
    frame_delta = forward_u32_delta(
        current.frame_sequence,
        previous.frame_sequence,
        "frame sequence",
    )
    capture_delta = forward_u32_delta(
        current.capture_time_us,
        previous.capture_time_us,
        "capture time",
    )
    forward_u32_delta(
        current.telemetry_time_us,
        previous.telemetry_time_us,
        "telemetry time",
    )
    return bool(
        frame_delta != 1
        or capture_delta < timing.minimum_period_us
        or capture_delta > timing.maximum_period_us
        or current.mission_epoch != previous.mission_epoch
        or current.arming_epoch != previous.arming_epoch
        or current.target_id != previous.target_id
    )


def validate_edge_proposal_stamp(
    proposal: EdgeProposalStamp,
    latest_input: EdgeInputStamp,
    *,
    previous_proposal_sequence: int | None,
    stm32_now_us: int,
    timing: EdgeTimingProfile,
    state_reset_applied: bool,
    input_reset_required: bool,
) -> int:
    if (
        type(state_reset_applied) is not bool
        or type(input_reset_required) is not bool
        or state_reset_applied != input_reset_required
    ):
        raise ValueError(
            "edge proposal reset acknowledgement does not match the input requirement"
        )
    expected = (
        latest_input.frame_sequence,
        latest_input.capture_time_us,
        latest_input.mission_epoch,
        latest_input.arming_epoch,
        latest_input.target_id,
    )
    actual = (
        proposal.source_frame_sequence,
        proposal.source_capture_time_us,
        proposal.mission_epoch,
        proposal.arming_epoch,
        proposal.source_target_id,
    )
    if actual != expected:
        raise ValueError("edge proposal does not echo the latest input authority")
    if previous_proposal_sequence is not None:
        forward_u32_delta(
            proposal.proposal_sequence,
            previous_proposal_sequence,
            "proposal sequence",
        )
    now = validate_u32(stm32_now_us, "STM32 current time")
    age = (now - proposal.source_capture_time_us) & 0xFFFF_FFFF
    if age >= UINT32_HALF_RANGE or age > timing.max_proposal_age_us:
        raise ValueError("edge proposal is stale or from the future")
    return age


def edge_timing_payload(
    timing: EdgeTimingProfile | None,
) -> dict[str, Any]:
    if timing is not None:
        return timing.payload()
    return {
        "status": "unmeasured_blocker",
        "clock_domain": "stm32_monotonic_uint32_us",
        "deployment_timing_authority": False,
        "required_measurement": (
            "measure the complete gray4 capture, telemetry alignment, CPX, "
            "inference, and proposal-return path"
        ),
        "required_bound_fields": [
            "nominal_period_us",
            "minimum_period_us",
            "maximum_period_us",
            "max_frame_telemetry_skew_us",
            "max_proposal_age_us",
            "measurement_frames",
            "measurement_sha256",
        ],
    }


def edge_wire_contract() -> dict[str, Any]:
    return {
        "policy_input": {
            "bytes": EDGE_INPUT_PACKET_BYTES,
            "endianness": "little",
            "packing": "packed_no_padding",
            "fields": [
                {"name": "protocol_version", "offset": 0, "dtype": "uint8", "bytes": 1, "value": 3},
                {"name": "flags", "offset": 1, "dtype": "uint8", "bytes": 1, "bit_0": "reset_state", "other_bits": "must_be_zero"},
                {"name": "frame_sequence", "offset": 2, "dtype": "uint32_le", "bytes": 4, "rule": "0 < ((new - old) & 0xffffffff) < 0x80000000"},
                {"name": "capture_time_us", "offset": 6, "dtype": "uint32_le", "bytes": 4, "clock": "runtime.timing.clock_domain", "rule": "0 < ((new - old) & 0xffffffff) < 0x80000000"},
                {"name": "telemetry_time_us", "offset": 10, "dtype": "uint32_le", "bytes": 4, "clock": "runtime.timing.clock_domain"},
                {"name": "mission_epoch", "offset": 14, "dtype": "uint32_le", "bytes": 4},
                {"name": "arming_epoch", "offset": 18, "dtype": "uint32_le", "bytes": 4},
                {"name": "target_id", "offset": 22, "dtype": "uint8", "bytes": 1, "allowed_ids": list(range(EDGE_MISSION_TOKEN_COUNT))},
                {"name": "telemetry", "offset": 23, "dtype": "float32_le[19]", "bytes": 76, "finite": True},
                {"name": "current_gray4", "offset": 99, "dtype": "uint8[1536]", "bytes": 1_536},
            ],
        },
        "policy_output": {
            "bytes": EDGE_OUTPUT_PACKET_BYTES,
            "endianness": "little",
            "packing": "packed_no_padding",
            "fields": [
                {"name": "protocol_version", "offset": 0, "dtype": "uint8", "bytes": 1, "value": 3},
                {"name": "flags", "offset": 1, "dtype": "uint8", "bytes": 1, "bit_0": "state_reset_applied", "other_bits": "must_be_zero"},
                {"name": "source_frame_sequence", "offset": 2, "dtype": "uint32_le", "bytes": 4},
                {"name": "source_capture_time_us", "offset": 6, "dtype": "uint32_le", "bytes": 4},
                {"name": "mission_epoch", "offset": 10, "dtype": "uint32_le", "bytes": 4},
                {"name": "arming_epoch", "offset": 14, "dtype": "uint32_le", "bytes": 4},
                {"name": "source_target_id", "offset": 18, "dtype": "uint8", "bytes": 1, "allowed_ids": list(range(EDGE_MISSION_TOKEN_COUNT))},
                {"name": "proposal_sequence", "offset": 19, "dtype": "uint32_le", "bytes": 4},
                {"name": "normalized_action", "offset": 23, "dtype": "float32_le[4]", "bytes": 16, "finite": True, "range": [-1.0, 1.0]},
            ],
        },
    }


def _absolute_u32_distance(first: int, second: int) -> int:
    forward = (first - second) & 0xFFFF_FFFF
    reverse = (second - first) & 0xFFFF_FFFF
    return min(forward, reverse)
