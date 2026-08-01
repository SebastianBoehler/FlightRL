from __future__ import annotations

from io import BytesIO
import struct

import numpy as np
from PIL import Image


AIDECK_IMAGE_MAGIC = 0xBC
AIDECK_RAW_FORMAT = 0
AIDECK_JPEG_FORMAT = 1
AIDECK_GRAY4_FORMAT = 2
AIDECK_IMAGE_HEADER = struct.Struct("<BHHBBI")
AIDECK_CPX_HEADER = struct.Struct("<HBB")
AIDECK_UDP_HANDSHAKE = b"FER"
MAX_AIDECK_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_AIDECK_PIXELS = 2048 * 2048
MAX_AIDECK_DEPTH = 4


def parse_cpx_packet(packet: bytes) -> tuple[int, int, bytes]:
    if len(packet) < AIDECK_CPX_HEADER.size:
        raise ValueError(f"AI Deck packet is too short: {len(packet)}")
    advertised_length, routing, function = AIDECK_CPX_HEADER.unpack_from(packet)
    if advertised_length + 2 != len(packet):
        raise ValueError(
            f"AI Deck packet length mismatch: advertised={advertised_length + 2} actual={len(packet)}"
        )
    return routing, function, packet[AIDECK_CPX_HEADER.size :]


def parse_image_header(payload: bytes) -> tuple[int, int, int, int, int]:
    if len(payload) != AIDECK_IMAGE_HEADER.size:
        raise ValueError(f"invalid AI Deck image header size {len(payload)}")
    magic, width, height, depth, image_format, size = AIDECK_IMAGE_HEADER.unpack(payload)
    if magic != AIDECK_IMAGE_MAGIC:
        raise ValueError(f"invalid AI Deck image magic 0x{magic:02x}")
    validate_image_metadata(width, height, depth, image_format, size)
    return width, height, depth, image_format, size


def try_image_header(payload: bytes) -> tuple[int, int, int, int, int] | None:
    if len(payload) != AIDECK_IMAGE_HEADER.size or payload[0] != AIDECK_IMAGE_MAGIC:
        return None
    return parse_image_header(payload)


def validate_image_metadata(width: int, height: int, depth: int, image_format: int, size: int) -> None:
    if width <= 0 or height <= 0 or depth <= 0 or size <= 0:
        raise ValueError("AI Deck image header contains non-positive dimensions")
    if depth > MAX_AIDECK_DEPTH:
        raise ValueError(f"AI Deck image depth {depth} exceeds {MAX_AIDECK_DEPTH}")
    pixel_count = int(width) * int(height)
    if pixel_count > MAX_AIDECK_PIXELS:
        raise ValueError(f"AI Deck image contains {pixel_count} pixels; limit is {MAX_AIDECK_PIXELS}")
    if size > MAX_AIDECK_PAYLOAD_BYTES:
        raise ValueError(f"AI Deck payload advertises {size} bytes; limit is {MAX_AIDECK_PAYLOAD_BYTES}")
    if image_format == AIDECK_RAW_FORMAT:
        expected = pixel_count * depth
        if size != expected:
            raise ValueError(f"AI Deck raw payload advertises {size} bytes; expected {expected}")
    elif image_format == AIDECK_GRAY4_FORMAT:
        expected = (pixel_count * depth + 1) // 2
        if size != expected:
            raise ValueError(f"AI Deck gray4 payload advertises {size} bytes; expected {expected}")
    elif image_format != AIDECK_JPEG_FORMAT:
        raise ValueError(f"unsupported AI Deck image format {image_format}")


def decode_pixels(payload: bytes, width: int, height: int, depth: int, image_format: int) -> np.ndarray:
    validate_image_metadata(width, height, depth, image_format, len(payload))
    if image_format == AIDECK_RAW_FORMAT:
        pixels = np.frombuffer(payload, dtype=np.uint8)
        return pixels.reshape(_pixel_shape(width, height, depth)).copy()
    if image_format == AIDECK_GRAY4_FORMAT:
        pixel_count = int(width) * int(height) * int(depth)
        packed = np.frombuffer(payload, dtype=np.uint8)
        pixels = np.empty(packed.size * 2, dtype=np.uint8)
        pixels[0::2] = packed >> 4
        pixels[1::2] = packed & 0x0F
        return (pixels[:pixel_count] * 17).reshape(_pixel_shape(width, height, depth))
    try:
        with Image.open(BytesIO(payload)) as image:
            pixels = np.asarray(image)
    except OSError as exc:
        raise ValueError("AI Deck JPEG payload could not be decoded") from exc
    expected_shape = _pixel_shape(width, height, depth)
    if pixels.shape != expected_shape or pixels.dtype != np.uint8:
        raise ValueError(
            f"AI Deck JPEG array {pixels.shape}/{pixels.dtype} does not match {expected_shape}/uint8"
        )
    return pixels.copy()


def _pixel_shape(width: int, height: int, depth: int) -> tuple[int, ...]:
    return (height, width) if depth == 1 else (height, width, depth)
