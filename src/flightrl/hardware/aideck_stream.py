from __future__ import annotations

from io import BytesIO
from dataclasses import dataclass
import socket
import struct
from time import monotonic, time
from typing import Callable, Iterator

import numpy as np
from PIL import Image


AIDECK_IMAGE_MAGIC = 0xBC
AIDECK_RAW_FORMAT = 0
AIDECK_JPEG_FORMAT = 1
AIDECK_GRAY4_FORMAT = 2
AIDECK_IMAGE_HEADER = struct.Struct("<BHHBBI")
AIDECK_CPX_HEADER = struct.Struct("<HBB")
AIDECK_UDP_HANDSHAKE = b"FER"


@dataclass(frozen=True, slots=True)
class AiDeckFrame:
    index: int
    host_time_s: float
    width: int
    height: int
    depth: int
    format: int
    pixels: np.ndarray


class AiDeckStream:
    def __init__(
        self,
        host: str = "192.168.4.1",
        port: int = 5000,
        *,
        timeout_s: float = 10.0,
        clock: Callable[[], float] = time,
        sock: socket.socket | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.clock = clock
        self._socket = sock
        self._owns_socket = sock is None
        self._index = 0

    def connect(self) -> None:
        if self._socket is None:
            self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        self._socket.settimeout(self.timeout_s)

    def close(self) -> None:
        if self._socket is not None and self._owns_socket:
            self._socket.close()
        self._socket = None

    def __enter__(self) -> AiDeckStream:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read_frame(self) -> AiDeckFrame:
        if self._socket is None:
            self.connect()
        assert self._socket is not None
        packet_length, _routing, _function = AIDECK_CPX_HEADER.unpack(_read_exact(self._socket, 4))
        if packet_length < 2:
            raise ValueError(f"invalid AI Deck image header packet length {packet_length}")
        header = _read_exact(self._socket, packet_length - 2)
        if len(header) != AIDECK_IMAGE_HEADER.size:
            raise ValueError(f"invalid AI Deck image header size {len(header)}")
        magic, width, height, depth, image_format, size = AIDECK_IMAGE_HEADER.unpack(header)
        host_time_s = self.clock()
        if magic != AIDECK_IMAGE_MAGIC:
            raise ValueError(f"invalid AI Deck image magic 0x{magic:02x}")

        payload = bytearray()
        while len(payload) < size:
            chunk_length, _destination, _source = AIDECK_CPX_HEADER.unpack(_read_exact(self._socket, 4))
            if chunk_length < 2:
                raise ValueError(f"invalid AI Deck image chunk length {chunk_length}")
            payload.extend(_read_exact(self._socket, chunk_length - 2))
        if len(payload) != size:
            raise ValueError(f"AI Deck payload exceeded advertised size: {len(payload)} != {size}")
        pixels = _decode_pixels(bytes(payload), width, height, depth, image_format)
        self._index += 1
        return AiDeckFrame(self._index, host_time_s, width, height, depth, image_format, pixels)

    def frames(self, limit: int | None = None) -> Iterator[AiDeckFrame]:
        count = 0
        while limit is None or count < limit:
            yield self.read_frame()
            count += 1


class AiDeckUdpStream:
    def __init__(
        self,
        host: str = "192.168.4.1",
        port: int = 5000,
        *,
        bind_host: str = "0.0.0.0",
        bind_port: int = 5001,
        timeout_s: float = 10.0,
        clock: Callable[[], float] = time,
        sock: socket.socket | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.bind_host = bind_host
        self.bind_port = int(bind_port)
        self.timeout_s = float(timeout_s)
        self.clock = clock
        self._socket = sock
        self._owns_socket = sock is None
        self._index = 0
        self.dropped_frames = 0
        self._connected = False

    def connect(self) -> None:
        if self._socket is None:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.bind((self.bind_host, self.bind_port))
        self._socket.settimeout(self.timeout_s)
        self._socket.sendto(AIDECK_UDP_HANDSHAKE, (self.host, self.port))
        self._connected = True

    def close(self) -> None:
        if self._socket is not None and self._owns_socket:
            self._socket.close()
        self._socket = None
        self._connected = False

    def __enter__(self) -> AiDeckUdpStream:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read_frame(self) -> AiDeckFrame:
        if not self._connected:
            self.connect()
        assert self._socket is not None
        deadline = monotonic() + self.timeout_s
        metadata: tuple[int, int, int, int, int] | None = None
        host_time_s = 0.0
        payload = bytearray()

        while True:
            remaining_s = deadline - monotonic()
            if remaining_s <= 0:
                raise TimeoutError("AI Deck UDP frame timed out")
            self._socket.settimeout(remaining_s)
            packet, _address = self._socket.recvfrom(2048)
            cpx_payload = _udp_cpx_payload(packet)
            header = _try_image_header(cpx_payload)
            if header is not None:
                if metadata is not None and len(payload) < metadata[-1]:
                    self.dropped_frames += 1
                metadata = header
                host_time_s = self.clock()
                payload.clear()
                continue
            if metadata is None:
                continue

            payload.extend(cpx_payload)
            width, height, depth, image_format, size = metadata
            if len(payload) < size:
                continue
            if len(payload) > size:
                self.dropped_frames += 1
                metadata = None
                payload.clear()
                continue
            try:
                pixels = _decode_pixels(bytes(payload), width, height, depth, image_format)
            except ValueError:
                self.dropped_frames += 1
                metadata = None
                payload.clear()
                continue
            self._index += 1
            return AiDeckFrame(self._index, host_time_s, width, height, depth, image_format, pixels)

    def frames(self, limit: int | None = None) -> Iterator[AiDeckFrame]:
        count = 0
        while limit is None or count < limit:
            yield self.read_frame()
            count += 1


def _read_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError(f"AI Deck stream closed after {len(data)}/{size} bytes")
        data.extend(chunk)
    return bytes(data)


def _udp_cpx_payload(packet: bytes) -> bytes:
    if len(packet) < AIDECK_CPX_HEADER.size:
        raise ValueError(f"AI Deck UDP packet is too short: {len(packet)}")
    advertised_length, _routing, _function = AIDECK_CPX_HEADER.unpack_from(packet)
    if advertised_length + 2 != len(packet):
        raise ValueError(
            f"AI Deck UDP packet length mismatch: advertised={advertised_length + 2} actual={len(packet)}"
        )
    return packet[AIDECK_CPX_HEADER.size :]


def _try_image_header(payload: bytes) -> tuple[int, int, int, int, int] | None:
    if len(payload) != AIDECK_IMAGE_HEADER.size or payload[0] != AIDECK_IMAGE_MAGIC:
        return None
    _magic, width, height, depth, image_format, size = AIDECK_IMAGE_HEADER.unpack(payload)
    if width <= 0 or height <= 0 or depth <= 0 or size <= 0:
        raise ValueError("AI Deck image header contains non-positive dimensions")
    return width, height, depth, image_format, size


def _decode_pixels(payload: bytes, width: int, height: int, depth: int, image_format: int) -> np.ndarray:
    if image_format == AIDECK_RAW_FORMAT:
        expected_size = int(width) * int(height) * int(depth)
        if len(payload) != expected_size:
            raise ValueError(f"AI Deck raw payload size {len(payload)} does not match {width}x{height}x{depth}")
        pixels = np.frombuffer(payload, dtype=np.uint8)
        return pixels.reshape((height, width) if depth == 1 else (height, width, depth)).copy()
    if image_format == AIDECK_JPEG_FORMAT:
        try:
            with Image.open(BytesIO(payload)) as image:
                pixels = np.asarray(image)
        except OSError as exc:
            raise ValueError("AI Deck JPEG payload could not be decoded") from exc
        if pixels.shape[:2] != (height, width):
            raise ValueError(f"AI Deck JPEG shape {pixels.shape} does not match {width}x{height}")
        return pixels.copy()
    if image_format == AIDECK_GRAY4_FORMAT:
        pixel_count = int(width) * int(height) * int(depth)
        expected_size = (pixel_count + 1) // 2
        if len(payload) != expected_size:
            raise ValueError(f"AI Deck gray4 payload size {len(payload)} does not match {width}x{height}x{depth}")
        packed = np.frombuffer(payload, dtype=np.uint8)
        pixels = np.empty(packed.size * 2, dtype=np.uint8)
        pixels[0::2] = packed >> 4
        pixels[1::2] = packed & 0x0F
        pixels = pixels[:pixel_count] * 17
        shape = (height, width) if depth == 1 else (height, width, depth)
        return pixels.reshape(shape)
    raise ValueError(f"unsupported AI Deck image format {image_format}")
