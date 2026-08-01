from __future__ import annotations

import argparse
import json
import math
from numbers import Integral, Real
from pathlib import Path
from time import monotonic

import numpy as np
from PIL import Image

from flightrl.hardware.aideck_stream import AiDeckStream, AiDeckUdpStream


CAPTURE_SCHEMA = "flightrl.aideck_decoded_frame_capture.v2"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture decoded AI Deck frames without loading or running a policy."
    )
    parser.add_argument("--transport", choices=("tcp", "udp"), default="tcp")
    parser.add_argument("--host", default="192.168.4.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--bind-port", type=int, default=5001)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ai_deck/decoded_frames.npz"),
    )
    parser.add_argument("--frame-dir", type=Path, help="save each decoded frame as PNG")
    args = parser.parse_args()

    validate_args(args, parser)
    frames: list[np.ndarray] = []
    host_times: list[float] = []
    frame_paths: list[str] = []
    capture_error: Exception | None = None
    stream = stream_from_args(args)
    start = monotonic()

    if args.frame_dir is not None:
        args.frame_dir.mkdir(parents=True, exist_ok=True)
    try:
        with stream:
            for frame in stream.frames(limit=args.frames):
                pixels = validate_frame(frame, len(frames) + 1, host_times[-1] if host_times else None)
                frames.append(pixels.copy())
                host_times.append(float(frame.host_time_s))
                if args.frame_dir is not None:
                    frame_path = args.frame_dir / f"frame-{frame.index:06d}.png"
                    Image.fromarray(pixels).save(frame_path)
                    frame_paths.append(str(frame_path))
    except (ConnectionError, OSError, TimeoutError, ValueError) as exc:
        capture_error = exc

    if not frames:
        raise SystemExit(f"AI Deck capture failed before the first complete frame: {capture_error}")
    try:
        decoded_frames = np.stack(frames)
    except ValueError as exc:
        raise SystemExit("AI Deck capture produced inconsistent decoded frame shapes") from exc
    if capture_error is None and len(frames) != args.frames:
        capture_error = RuntimeError(f"capture stopped after {len(frames)}/{args.frames} frames")
    complete = capture_error is None
    dropped_frames = validated_counter(stream, "dropped_frames")
    rejected_datagrams = validated_counter(stream, "rejected_datagrams")
    metadata = capture_metadata(args, decoded_frames, complete, dropped_frames, rejected_datagrams)
    metadata_json = json.dumps(metadata, sort_keys=True, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        decoded_frames=decoded_frames,
        host_time_s=np.asarray(host_times, dtype=np.float64),
        frame_paths=np.asarray(frame_paths, dtype=str),
        metadata_json=np.asarray(metadata_json),
        complete=np.asarray(complete),
        capture_error=np.asarray("" if capture_error is None else str(capture_error)),
        dropped_frames=np.asarray(dropped_frames),
        rejected_datagrams=np.asarray(rejected_datagrams),
    )
    provenance_path = args.output.with_suffix(args.output.suffix + ".provenance.json")
    provenance_path.write_text(metadata_json + "\n")
    if args.frame_dir is not None:
        frame_integrity = {
            "version": 1,
            "datasets": [
                {
                    "path": ".",
                    "status": "unreviewed",
                    "evidence": metadata["authority_reason"],
                }
            ],
        }
        (args.frame_dir / "frame-integrity.json").write_text(
            json.dumps(frame_integrity, indent=2, sort_keys=True) + "\n"
        )
    elapsed = monotonic() - start
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise SystemExit("AI Deck capture duration was nonfinite or non-positive")
    print(f"wrote {len(frames)} decoded frames to {args.output}")
    print(f"shape={decoded_frames.shape[1:]} rate_hz={len(frames) / elapsed:.2f}")
    if capture_error is not None:
        raise SystemExit(f"AI Deck capture ended early after {len(frames)} frames: {capture_error}")


def stream_from_args(args):
    if args.transport == "udp":
        return AiDeckUdpStream(
            args.host,
            args.port,
            bind_host=args.bind_host,
            bind_port=args.bind_port,
            timeout_s=args.timeout_s,
        )
    return AiDeckStream(args.host, args.port, timeout_s=args.timeout_s)


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0.0:
        parser.error("--timeout-s must be finite and positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if not 0 <= args.bind_port <= 65535:
        parser.error("--bind-port must be in [0, 65535]")
    if not args.host.strip():
        parser.error("--host must be non-empty")


def validate_frame(frame, expected_index: int, previous_time_s: float | None) -> np.ndarray:
    if isinstance(frame.index, bool) or not isinstance(frame.index, Integral) or frame.index != expected_index:
        raise ValueError(f"AI Deck frame index {frame.index} is not expected index {expected_index}")
    for label, value in (("width", frame.width), ("height", frame.height), ("depth", frame.depth)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"AI Deck frame {label} must be a positive integer")
    if isinstance(frame.format, bool) or not isinstance(frame.format, Integral) or frame.format not in {0, 1, 2}:
        raise ValueError("AI Deck frame format is invalid")
    if isinstance(frame.host_time_s, bool) or not isinstance(frame.host_time_s, Real):
        raise ValueError("AI Deck frame timestamp must be finite and non-negative")
    host_time_s = float(frame.host_time_s)
    if not math.isfinite(host_time_s) or host_time_s < 0.0:
        raise ValueError("AI Deck frame timestamp must be finite and non-negative")
    if previous_time_s is not None and host_time_s < previous_time_s:
        raise ValueError("AI Deck frame timestamps must be nondecreasing")
    pixels = np.asarray(frame.pixels)
    expected_shape = (frame.height, frame.width) if frame.depth == 1 else (frame.height, frame.width, frame.depth)
    if pixels.dtype != np.uint8 or pixels.shape != expected_shape:
        raise ValueError(
            f"AI Deck decoded frame {pixels.shape}/{pixels.dtype} does not match {expected_shape}/uint8"
        )
    return pixels


def validated_counter(stream, name: str) -> int:
    value = getattr(stream, name, 0)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise SystemExit(f"AI Deck stream reported invalid {name}={value!r}")
    return int(value)


def capture_metadata(
    args: argparse.Namespace,
    decoded_frames: np.ndarray,
    complete: bool,
    dropped_frames: int,
    rejected_datagrams: int,
) -> dict[str, object]:
    udp = args.transport == "udp"
    authority_reason = (
        "UDP capture is unreviewed because firmware provides no chunk sequence field or application frame checksum"
        if udp
        else "capture requires explicit frame-integrity review before any training use"
    )
    return {
        "schema": CAPTURE_SCHEMA,
        "transport": args.transport,
        "configured_source_endpoint": {"host": args.host, "port": args.port},
        "decoded_frame_shape": list(decoded_frames.shape[1:]),
        "decoded_frame_dtype": str(decoded_frames.dtype),
        "captured_frames": int(len(decoded_frames)),
        "requested_frames": int(args.frames),
        "complete": bool(complete),
        "dropped_frames": dropped_frames,
        "rejected_datagrams": rejected_datagrams,
        "policy_outputs_present": False,
        "edge_v3_preprocessing_applied": False,
        "integrity_status": "unreviewed",
        "training_authority": False,
        "deployment_authority": False,
        "authority_reason": authority_reason,
        "transport_integrity": {
            "source_endpoint_enforced": True,
            "cpx_route_function_consistency_enforced": True,
            "ordered_transport": not udp,
            "firmware_sequence_field_present": False,
            "chunk_order_verified": not udp,
            "application_frame_checksum_present": False,
            "udp_reassembly_authoritative": False,
            "udp_limitation": (
                "datagram ordering cannot be proven without firmware sequence numbers"
                if udp
                else "not_applicable"
            ),
        },
    }


if __name__ == "__main__":
    main()
