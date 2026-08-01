from __future__ import annotations

from dataclasses import dataclass
import struct

import torch

from flightrl.puffer4_edge_contract import (
    EDGE_FRAME_PIXELS,
    EDGE_OBSERVATION_DIM,
    edge_target_one_hot,
    validate_edge_target_id,
    validate_normalized_edge_action,
    validate_normalized_edge_telemetry,
)
from flightrl.puffer4_edge_wire import (
    EDGE_INPUT_PACKET_BYTES,
    EDGE_OUTPUT_PACKET_BYTES,
    EdgeInputStamp,
    EdgeProposalStamp,
)


EDGE_PROTOCOL_VERSION = 3
EDGE_RESET_FLAG = 1
_INPUT_HEADER = struct.Struct("<BBIIIII B 19f")
_OUTPUT_RECORD = struct.Struct("<BBIIIIBI4f")
_PACKED_FRAME_BYTES = EDGE_FRAME_PIXELS // 2
_TELEMETRY_VALUES = struct.Struct("<19f")
_ACTION_VALUES = struct.Struct("<4f")


@dataclass(frozen=True, slots=True)
class DecodedEdgeInput:
    stamp: EdgeInputStamp
    reset_requested: bool
    telemetry: tuple[float, ...]
    packed_gray4: bytes

    def model_observation(self) -> torch.Tensor:
        pixels = unpack_gray4(self.packed_gray4)
        values = (*self.telemetry, *edge_target_one_hot(self.stamp.target_id))
        suffix = torch.tensor(values, dtype=torch.float32)
        observation = torch.cat((pixels, suffix)).unsqueeze(0)
        if observation.shape != (1, EDGE_OBSERVATION_DIM):
            raise RuntimeError("decoded edge observation violates the model ABI")
        return observation


@dataclass(frozen=True, slots=True)
class DecodedEdgeProposal:
    stamp: EdgeProposalStamp
    state_reset_applied: bool
    action: tuple[float, ...]


def pack_gray4(frame: torch.Tensor) -> bytes:
    flat = frame.detach().to(device="cpu")
    if flat.ndim != 1 or flat.numel() != EDGE_FRAME_PIXELS:
        raise ValueError(f"edge frame must contain {EDGE_FRAME_PIXELS} pixels")
    if flat.dtype != torch.float32 or not bool(torch.isfinite(flat).all()):
        raise ValueError("edge frame must be finite float32")
    levels = flat * 15.0
    if bool(torch.any((flat < 0.0) | (flat > 1.0))) or not torch.allclose(
        levels,
        levels.round(),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError("edge frame must contain exact gray4 levels")
    nibbles = levels.round().to(torch.uint8)
    packed = (nibbles[0::2] << 4) | nibbles[1::2]
    return packed.numpy().tobytes()


def unpack_gray4(payload: bytes | bytearray | memoryview) -> torch.Tensor:
    raw = bytes(payload)
    if len(raw) != _PACKED_FRAME_BYTES:
        raise ValueError(f"packed edge frame must contain {_PACKED_FRAME_BYTES} bytes")
    packed = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    nibbles = torch.empty(EDGE_FRAME_PIXELS, dtype=torch.uint8)
    nibbles[0::2] = packed >> 4
    nibbles[1::2] = packed & 0x0F
    return nibbles.to(torch.float32) / 15.0


def encode_edge_input(
    stamp: EdgeInputStamp,
    *,
    reset_requested: bool,
    telemetry: object,
    packed_gray4: bytes | bytearray | memoryview,
) -> bytes:
    if type(reset_requested) is not bool:
        raise ValueError("edge reset flag must be boolean")
    normalized = validate_normalized_edge_telemetry(telemetry)
    normalized = validate_normalized_edge_telemetry(
        _TELEMETRY_VALUES.unpack(_TELEMETRY_VALUES.pack(*normalized))
    )
    frame = bytes(packed_gray4)
    if len(frame) != _PACKED_FRAME_BYTES:
        raise ValueError("packed edge frame length is invalid")
    header = _INPUT_HEADER.pack(
        EDGE_PROTOCOL_VERSION,
        EDGE_RESET_FLAG if reset_requested else 0,
        stamp.frame_sequence,
        stamp.capture_time_us,
        stamp.telemetry_time_us,
        stamp.mission_epoch,
        stamp.arming_epoch,
        stamp.target_id,
        *normalized,
    )
    packet = header + frame
    if len(packet) != EDGE_INPUT_PACKET_BYTES:
        raise RuntimeError("encoded edge input length violates the wire contract")
    return packet


def decode_edge_input(packet: bytes | bytearray | memoryview) -> DecodedEdgeInput:
    raw = bytes(packet)
    if len(raw) != EDGE_INPUT_PACKET_BYTES:
        raise ValueError("edge input packet length is invalid")
    values = _INPUT_HEADER.unpack_from(raw)
    version, flags = values[:2]
    if version != EDGE_PROTOCOL_VERSION or flags & ~EDGE_RESET_FLAG:
        raise ValueError("edge input version or flags are invalid")
    stamp = EdgeInputStamp(*values[2:8])
    telemetry = validate_normalized_edge_telemetry(values[8:])
    return DecodedEdgeInput(
        stamp=stamp,
        reset_requested=bool(flags & EDGE_RESET_FLAG),
        telemetry=telemetry,
        packed_gray4=raw[_INPUT_HEADER.size :],
    )


def encode_edge_proposal(
    stamp: EdgeProposalStamp,
    *,
    state_reset_applied: bool,
    action: object,
) -> bytes:
    if type(state_reset_applied) is not bool:
        raise ValueError("edge proposal reset flag must be boolean")
    normalized = validate_normalized_edge_action(action)
    normalized = validate_normalized_edge_action(
        _ACTION_VALUES.unpack(_ACTION_VALUES.pack(*normalized))
    )
    packet = _OUTPUT_RECORD.pack(
        EDGE_PROTOCOL_VERSION,
        EDGE_RESET_FLAG if state_reset_applied else 0,
        stamp.source_frame_sequence,
        stamp.source_capture_time_us,
        stamp.mission_epoch,
        stamp.arming_epoch,
        stamp.source_target_id,
        stamp.proposal_sequence,
        *normalized,
    )
    if len(packet) != EDGE_OUTPUT_PACKET_BYTES:
        raise RuntimeError("encoded edge proposal length violates the wire contract")
    return packet


def decode_edge_proposal(
    packet: bytes | bytearray | memoryview,
) -> DecodedEdgeProposal:
    raw = bytes(packet)
    if len(raw) != EDGE_OUTPUT_PACKET_BYTES:
        raise ValueError("edge proposal packet length is invalid")
    values = _OUTPUT_RECORD.unpack(raw)
    version, flags = values[:2]
    if version != EDGE_PROTOCOL_VERSION or flags & ~EDGE_RESET_FLAG:
        raise ValueError("edge proposal version or flags are invalid")
    stamp = EdgeProposalStamp(*values[2:8])
    action = validate_normalized_edge_action(values[8:])
    validate_edge_target_id(stamp.source_target_id)
    return DecodedEdgeProposal(stamp, bool(flags & EDGE_RESET_FLAG), action)
