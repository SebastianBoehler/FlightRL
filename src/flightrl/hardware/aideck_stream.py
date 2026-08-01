from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
import socket
from time import monotonic, time
from typing import Callable, Iterator

import numpy as np

from .aideck_protocol import (
    AIDECK_CPX_HEADER,
    AIDECK_UDP_HANDSHAKE,
    decode_pixels,
    parse_image_header,
    parse_cpx_packet,
    try_image_header,
)


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
        _validate_stream_config(host, port, timeout_s)
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
        packet_length, routing, function = AIDECK_CPX_HEADER.unpack(_read_exact(self._socket, 4))
        if packet_length < 2:
            raise ValueError(f"invalid AI Deck image header packet length {packet_length}")
        header = _read_exact(self._socket, packet_length - 2)
        width, height, depth, image_format, size = parse_image_header(header)
        host_time_s = _finite_host_time(self.clock())

        payload = bytearray()
        while len(payload) < size:
            chunk_length, chunk_routing, chunk_function = AIDECK_CPX_HEADER.unpack(_read_exact(self._socket, 4))
            if chunk_length < 2:
                raise ValueError(f"invalid AI Deck image chunk length {chunk_length}")
            if (chunk_routing, chunk_function) != (routing, function):
                raise ValueError("AI Deck TCP frame changed CPX routing/function mid-frame")
            chunk = _read_exact(self._socket, chunk_length - 2)
            if len(chunk) > size - len(payload):
                raise ValueError(f"AI Deck payload exceeded advertised size {size}")
            payload.extend(chunk)
        pixels = decode_pixels(bytes(payload), width, height, depth, image_format)
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
        _validate_stream_config(host, port, timeout_s)
        if isinstance(bind_port, bool) or not isinstance(bind_port, Integral) or not 0 <= bind_port <= 65535:
            raise ValueError("AI Deck UDP bind port must be in [0, 65535]")
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
        self.rejected_datagrams = 0
        self._connected = False
        self._peer_ip: str | None = None

    def connect(self) -> None:
        if self._socket is None:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.bind((self.bind_host, self.bind_port))
        try:
            self._peer_ip = socket.gethostbyname(self.host)
        except OSError as exc:
            raise ConnectionError(f"could not resolve AI Deck UDP host {self.host!r}") from exc
        self._socket.settimeout(self.timeout_s)
        self._socket.sendto(AIDECK_UDP_HANDSHAKE, (self._peer_ip, self.port))
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
        stream_key: tuple[int, int] | None = None
        host_time_s = 0.0
        payload = bytearray()

        while True:
            remaining_s = deadline - monotonic()
            if remaining_s <= 0:
                raise TimeoutError("AI Deck UDP frame timed out")
            self._socket.settimeout(remaining_s)
            packet, address = self._socket.recvfrom(2048)
            if not self._source_matches(address):
                self.rejected_datagrams += 1
                continue
            routing, function, cpx_payload = parse_cpx_packet(packet)
            header = try_image_header(cpx_payload)
            if header is not None:
                if metadata is not None and len(payload) < metadata[-1]:
                    self.dropped_frames += 1
                metadata = header
                stream_key = (routing, function)
                host_time_s = _finite_host_time(self.clock())
                payload.clear()
                continue
            if metadata is None:
                continue
            if (routing, function) != stream_key:
                self.rejected_datagrams += 1
                continue

            payload.extend(cpx_payload)
            width, height, depth, image_format, size = metadata
            if len(payload) < size:
                continue
            if len(payload) > size:
                self.dropped_frames += 1
                metadata = None
                stream_key = None
                payload.clear()
                continue
            try:
                pixels = decode_pixels(bytes(payload), width, height, depth, image_format)
            except ValueError:
                self.dropped_frames += 1
                metadata = None
                stream_key = None
                payload.clear()
                continue
            self._index += 1
            return AiDeckFrame(self._index, host_time_s, width, height, depth, image_format, pixels)

    def _source_matches(self, address: tuple[object, ...]) -> bool:
        return (
            self._peer_ip is not None
            and len(address) >= 2
            and address[0] == self._peer_ip
            and address[1] == self.port
        )

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


def _validate_stream_config(host: str, port: int, timeout_s: float) -> None:
    if not isinstance(host, str) or not host.strip():
        raise ValueError("AI Deck host must be a non-empty string")
    if isinstance(port, bool) or not isinstance(port, Integral) or not 1 <= port <= 65535:
        raise ValueError("AI Deck port must be in [1, 65535]")
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, Real)
        or not math.isfinite(float(timeout_s))
        or timeout_s <= 0.0
    ):
        raise ValueError("AI Deck timeout must be finite and positive")


def _finite_host_time(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("AI Deck frame host timestamp must be finite and non-negative")
    timestamp = float(value)
    if not math.isfinite(timestamp) or timestamp < 0.0:
        raise ValueError("AI Deck frame host timestamp must be finite and non-negative")
    return timestamp
