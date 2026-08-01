from __future__ import annotations

from io import BytesIO
import socket
import struct

import numpy as np
import pytest
from PIL import Image

from flightrl.hardware.aideck_protocol import MAX_AIDECK_PAYLOAD_BYTES
from flightrl.hardware.aideck_stream import AiDeckStream, AiDeckUdpStream


def test_stream_decodes_raw_frame_protocol() -> None:
    receiver, sender = socket.socketpair()
    pixels = bytes([1, 2, 3, 4, 5, 6])
    sender.sendall(_frame_packet(width=3, height=2, depth=1, image_format=0, payload=pixels))
    stream = AiDeckStream(sock=receiver, clock=lambda: 123.5)

    try:
        frame = stream.read_frame()
    finally:
        receiver.close()
        sender.close()

    assert frame.index == 1
    assert frame.host_time_s == 123.5
    assert frame.pixels.shape == (2, 3)
    assert np.array_equal(frame.pixels, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8))


def test_stream_decodes_jpeg_frame() -> None:
    receiver, sender = socket.socketpair()
    jpeg = _jpeg_bytes(np.full((2, 3), 80, dtype=np.uint8))
    sender.sendall(_frame_packet(width=3, height=2, depth=1, image_format=1, payload=jpeg))
    stream = AiDeckStream(sock=receiver)

    try:
        frame = stream.read_frame()
    finally:
        receiver.close()
        sender.close()

    assert frame.pixels.shape == (2, 3)
    assert np.allclose(frame.pixels, 80, atol=2)


def test_stream_decodes_packed_gray4_frame() -> None:
    receiver, sender = socket.socketpair()
    sender.sendall(_frame_packet(width=3, height=1, depth=1, image_format=2, payload=bytes([0x1A, 0xF0])))
    stream = AiDeckStream(sock=receiver)

    try:
        frame = stream.read_frame()
    finally:
        receiver.close()
        sender.close()

    assert np.array_equal(frame.pixels, np.array([[17, 170, 255]], dtype=np.uint8))


def test_udp_stream_reassembles_jpeg_after_handshake() -> None:
    receiver = _FakeUdpSocket()
    jpeg = _jpeg_bytes(np.full((2, 3), 120, dtype=np.uint8))
    receiver.packets.extend(
        [
            (_udp_packet(struct.pack("<BHHBBI", 0xBC, 3, 2, 1, 1, len(jpeg))), ("192.168.4.1", 5000)),
            (_udp_packet(jpeg[:5]), ("192.168.4.1", 5000)),
            (_udp_packet(jpeg[5:]), ("192.168.4.1", 5000)),
        ]
    )
    stream = AiDeckUdpStream(sock=receiver, clock=lambda: 456.0)

    frame = stream.read_frame()

    assert receiver.sent == [(b"FER", ("192.168.4.1", 5000))]
    assert frame.host_time_s == 456.0
    assert frame.pixels.shape == (2, 3)
    assert np.allclose(frame.pixels, 120, atol=2)


def test_tcp_rejects_oversized_payload_before_reading_body() -> None:
    receiver, sender = socket.socketpair()
    header = struct.pack("<BHHBBI", 0xBC, 3, 2, 1, 1, MAX_AIDECK_PAYLOAD_BYTES + 1)
    sender.sendall(struct.pack("<HBB", len(header) + 2, 0, 0) + header)
    stream = AiDeckStream(sock=receiver)

    try:
        with pytest.raises(ValueError, match="payload advertises"):
            stream.read_frame()
    finally:
        receiver.close()
        sender.close()


def test_tcp_rejects_inconsistent_raw_size_before_reading_body() -> None:
    receiver, sender = socket.socketpair()
    header = struct.pack("<BHHBBI", 0xBC, 2, 2, 1, 0, 3)
    sender.sendall(struct.pack("<HBB", len(header) + 2, 0, 0) + header)
    stream = AiDeckStream(sock=receiver)

    try:
        with pytest.raises(ValueError, match="expected 4"):
            stream.read_frame()
    finally:
        receiver.close()
        sender.close()


def test_tcp_rejects_nonfinite_timestamp_and_cpx_stream_change() -> None:
    receiver, sender = socket.socketpair()
    sender.sendall(_frame_packet(width=1, height=1, depth=1, image_format=0, payload=b"\x01"))
    stream = AiDeckStream(sock=receiver, clock=lambda: float("nan"))
    try:
        with pytest.raises(ValueError, match="timestamp must be finite"):
            stream.read_frame()
    finally:
        receiver.close()
        sender.close()

    receiver, sender = socket.socketpair()
    header = struct.pack("<BHHBBI", 0xBC, 1, 1, 1, 0, 1)
    sender.sendall(struct.pack("<HBB", len(header) + 2, 7, 9) + header)
    sender.sendall(struct.pack("<HBB", 3, 7, 8) + b"\x01")
    stream = AiDeckStream(sock=receiver)
    try:
        with pytest.raises(ValueError, match="routing/function"):
            stream.read_frame()
    finally:
        receiver.close()
        sender.close()


def test_udp_filters_source_and_cpx_stream_without_claiming_order() -> None:
    receiver = _FakeUdpSocket()
    jpeg = _jpeg_bytes(np.full((2, 3), 140, dtype=np.uint8))
    header = struct.pack("<BHHBBI", 0xBC, 3, 2, 1, 1, len(jpeg))
    receiver.packets.extend(
        [
            (_udp_packet(header, routing=7, function=9), ("192.168.4.2", 5000)),
            (_udp_packet(header, routing=7, function=9), ("192.168.4.1", 5000)),
            (_udp_packet(b"not-image-data", routing=7, function=8), ("192.168.4.1", 5000)),
            (_udp_packet(jpeg[:5], routing=7, function=9), ("192.168.4.1", 5000)),
            (_udp_packet(jpeg[5:], routing=7, function=9), ("192.168.4.1", 5000)),
        ]
    )
    stream = AiDeckUdpStream(sock=receiver)

    frame = stream.read_frame()

    assert frame.pixels.shape == (2, 3)
    assert stream.rejected_datagrams == 2
    assert stream.dropped_frames == 0


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_stream_rejects_invalid_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        AiDeckStream(timeout_s=timeout)


def _frame_packet(*, width: int, height: int, depth: int, image_format: int, payload: bytes) -> bytes:
    image_header = struct.pack("<BHHBBI", 0xBC, width, height, depth, image_format, len(payload))
    header_packet = struct.pack("<HBB", len(image_header) + 2, 0, 0) + image_header
    chunk_packet = struct.pack("<HBB", len(payload) + 2, 0, 0) + payload
    return header_packet + chunk_packet


def _udp_packet(payload: bytes, *, routing: int = 0, function: int = 0) -> bytes:
    return struct.pack("<HBB", len(payload) + 2, routing, function) + payload


def _jpeg_bytes(pixels: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(pixels).save(output, format="JPEG")
    return output.getvalue()


class _FakeUdpSocket:
    def __init__(self) -> None:
        self.packets: list[tuple[bytes, tuple[str, int]]] = []
        self.sent: list[tuple[bytes, tuple[str, int]]] = []

    def settimeout(self, _timeout: float) -> None:
        pass

    def sendto(self, payload: bytes, address: tuple[str, int]) -> None:
        self.sent.append((payload, address))

    def recvfrom(self, _size: int) -> tuple[bytes, tuple[str, int]]:
        return self.packets.pop(0)
