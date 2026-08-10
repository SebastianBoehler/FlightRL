from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from flightrl.hardware.aideck_protocol import AIDECK_GRAY4_FORMAT, decode_pixels
from flightrl.puffer4_edge_contract import EDGE_FRAME_PIXELS
from flightrl.puffer4_edge_wire import (
    EDGE_INPUT_PACKET_BYTES,
    EDGE_OUTPUT_PACKET_BYTES,
    EdgeInputStamp,
    EdgeProposalStamp,
)
from flightrl.puffer4_edge_wire_codec import (
    decode_edge_input,
    decode_edge_proposal,
    encode_edge_input,
    encode_edge_proposal,
    pack_gray4,
    unpack_gray4,
)


def _telemetry() -> list[float]:
    values = [0.0] * 19
    values[8] = 1.0
    values[14] = 1.0
    return values


def test_gray4_codec_uses_even_high_odd_low_nibbles() -> None:
    levels = torch.arange(EDGE_FRAME_PIXELS, dtype=torch.int64) % 16
    frame = levels.to(torch.float32) / 15.0

    packed = pack_gray4(frame)
    decoded = unpack_gray4(packed)

    assert packed[:4] == bytes((0x01, 0x23, 0x45, 0x67))
    assert len(packed) == EDGE_FRAME_PIXELS // 2
    assert torch.equal(decoded, frame)


def test_aideck_decoded_gray4_needs_only_divide_by_255_for_edge_codec() -> None:
    payload = np.arange(EDGE_FRAME_PIXELS // 2, dtype=np.uint8).tobytes()
    decoded = decode_pixels(payload, 64, 48, 1, AIDECK_GRAY4_FORMAT)
    visual_segment = (
        torch.from_numpy(decoded.copy()).reshape(-1).to(torch.float32) / 255.0
    )

    repacked = pack_gray4(visual_segment)

    assert repacked == payload
    assert torch.equal(unpack_gray4(repacked), visual_segment)


def test_input_codec_matches_canonical_offsets_and_model_observation() -> None:
    stamp = EdgeInputStamp(7, 100, 99, 3, 4, 0)
    frame = torch.zeros(EDGE_FRAME_PIXELS, dtype=torch.float32)
    packet = encode_edge_input(
        stamp,
        reset_requested=True,
        telemetry=_telemetry(),
        packed_gray4=pack_gray4(frame),
    )

    decoded = decode_edge_input(packet)

    assert len(packet) == EDGE_INPUT_PACKET_BYTES
    assert packet[:2] == b"\x03\x01"
    assert struct.unpack_from("<I", packet, 2)[0] == 7
    assert packet[22] == 0
    assert decoded.stamp == stamp
    assert decoded.reset_requested is True
    assert decoded.model_observation().shape == (1, 3094)
    assert decoded.model_observation()[0, -3:].tolist() == [1.0, 0.0, 0.0]


def test_proposal_codec_matches_canonical_offsets() -> None:
    stamp = EdgeProposalStamp(7, 100, 3, 4, 0, 9)
    packet = encode_edge_proposal(
        stamp,
        state_reset_applied=False,
        action=(0.25, 0.0, 0.0, -0.5),
    )

    decoded = decode_edge_proposal(packet)

    assert len(packet) == EDGE_OUTPUT_PACKET_BYTES
    assert packet[:2] == b"\x03\x00"
    assert packet[18] == 0
    assert struct.unpack_from("<I", packet, 19)[0] == 9
    assert decoded.stamp == stamp
    assert decoded.action == pytest.approx((0.25, 0.0, 0.0, -0.5))


@pytest.mark.parametrize("size", (0, EDGE_INPUT_PACKET_BYTES - 1, EDGE_INPUT_PACKET_BYTES + 1))
def test_input_decoder_rejects_wrong_length(size: int) -> None:
    with pytest.raises(ValueError, match="length"):
        decode_edge_input(bytes(size))


def test_codecs_reject_reserved_flags_and_nonfinite_values() -> None:
    stamp = EdgeInputStamp(7, 100, 99, 3, 4, 0)
    packet = bytearray(
        encode_edge_input(
            stamp,
            reset_requested=True,
            telemetry=_telemetry(),
            packed_gray4=bytes(1536),
        )
    )
    packet[1] = 2
    with pytest.raises(ValueError, match="flags"):
        decode_edge_input(packet)

    invalid = _telemetry()
    invalid[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        encode_edge_input(
            stamp,
            reset_requested=True,
            telemetry=invalid,
            packed_gray4=bytes(1536),
        )


def test_encoder_never_emits_float32_telemetry_its_decoder_rejects() -> None:
    telemetry = _telemetry()
    telemetry[13] = 0.99990001
    telemetry[14] = 0.0

    with pytest.raises(ValueError, match="yaw"):
        encode_edge_input(
            EdgeInputStamp(7, 100, 99, 3, 4, 0),
            reset_requested=True,
            telemetry=telemetry,
            packed_gray4=bytes(1536),
        )
