from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic, perf_counter

import numpy as np
from PIL import Image

from flightrl.hardware.aideck_stream import AiDeckStream, AiDeckUdpStream
from flightrl.vision import (
    VisionObservationConfig,
    VisionObservationEncoder,
    load_vision_action_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture AI Deck frames through the FlightRL vision contract.")
    parser.add_argument("--transport", choices=("tcp", "udp"), default="tcp")
    parser.add_argument("--host", default="192.168.4.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--bind-port", type=int, default=5001)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ai_deck/vision_observations.npz"))
    parser.add_argument("--frame-dir", type=Path, help="save each raw source frame as a PNG")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--color-mode", choices=("grayscale", "rgb"), default="grayscale")
    parser.add_argument("--input-color-order", choices=("rgb", "bgr"), default="rgb")
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--include-delta", action="store_true")
    parser.add_argument("--include-motion-mask", action="store_true")
    parser.add_argument("--motion-threshold", type=float, default=0.08)
    parser.add_argument("--normalization", choices=("zero_one", "minus_one_one"), default="minus_one_one")
    parser.add_argument("--policy-checkpoint", type=Path, help="run a vision-action checkpoint in shadow mode")
    args = parser.parse_args()

    if args.frames <= 0:
        raise SystemExit("--frames must be positive")
    config = VisionObservationConfig(
        width=args.width,
        height=args.height,
        color_mode=args.color_mode,
        input_color_order=args.input_color_order,
        frame_stack=args.frame_stack,
        include_delta=args.include_delta,
        include_motion_mask=args.include_motion_mask,
        motion_threshold=args.motion_threshold,
        normalization=args.normalization,
    )
    encoder = VisionObservationEncoder(config)
    policy = load_capture_policy(args.policy_checkpoint, config)
    observations: list[np.ndarray] = []
    host_times: list[float] = []
    source_means: list[float] = []
    frame_paths: list[str] = []
    policy_actions: list[np.ndarray] = []
    policy_actions_physical: list[np.ndarray] = []
    policy_inference_ms: list[float] = []
    capture_error: Exception | None = None
    start = monotonic()
    stream = stream_from_args(args)

    if args.frame_dir is not None:
        args.frame_dir.mkdir(parents=True, exist_ok=True)
    try:
        with stream:
            for frame in stream.frames(limit=args.frames):
                observation = encoder.encode(frame.pixels)
                observations.append(observation)
                host_times.append(frame.host_time_s)
                source_means.append(float(frame.pixels.mean()))
                if policy is not None:
                    normalized, physical, inference_ms = infer_policy(policy, observation)
                    policy_actions.append(normalized)
                    policy_actions_physical.append(physical)
                    policy_inference_ms.append(inference_ms)
                if args.frame_dir is not None:
                    frame_path = args.frame_dir / f"frame-{frame.index:06d}.png"
                    Image.fromarray(frame.pixels).save(frame_path)
                    frame_paths.append(str(frame_path))
    except (ConnectionError, OSError, TimeoutError, ValueError) as exc:
        capture_error = exc

    elapsed = monotonic() - start
    dropped_frames = int(getattr(stream, "dropped_frames", 0))
    if not observations:
        raise SystemExit(f"AI Deck capture failed before the first complete frame: {capture_error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        observations=np.stack(observations),
        host_time_s=np.asarray(host_times, dtype=np.float64),
        source_mean=np.asarray(source_means, dtype=np.float32),
        frame_paths=np.asarray(frame_paths),
        contract_json=np.asarray(json.dumps(config.metadata(), sort_keys=True)),
        complete=np.asarray(capture_error is None),
        capture_error=np.asarray("" if capture_error is None else str(capture_error)),
        dropped_frames=np.asarray(dropped_frames),
        policy_actions=np.asarray(policy_actions, dtype=np.float32).reshape((-1, 3)),
        policy_actions_physical=np.asarray(policy_actions_physical, dtype=np.float32).reshape((-1, 3)),
        policy_inference_ms=np.asarray(policy_inference_ms, dtype=np.float32),
        policy_checkpoint=np.asarray("" if args.policy_checkpoint is None else str(args.policy_checkpoint)),
    )
    print(f"wrote {len(observations)} observations to {args.output}")
    print(f"shape={config.shape} flat_dim={config.flat_dim} rate_hz={len(observations) / elapsed:.2f}")
    print(f"dropped_frames={dropped_frames}")
    print(f"source_mean_min={min(source_means):.2f} source_mean_max={max(source_means):.2f}")
    if policy_inference_ms:
        print(
            f"policy_inference_ms_p50={np.percentile(policy_inference_ms, 50):.3f} "
            f"p95={np.percentile(policy_inference_ms, 95):.3f}"
        )
    if capture_error is not None:
        raise SystemExit(f"AI Deck capture ended early after {len(observations)} frames: {capture_error}")


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


def load_capture_policy(checkpoint: Path | None, config: VisionObservationConfig):
    if checkpoint is None:
        return None
    policy = load_vision_action_policy(checkpoint)
    expected = (policy.metadata.channels, policy.metadata.height, policy.metadata.width)
    if expected != config.shape:
        raise SystemExit(f"policy expects vision shape {expected}, capture contract produces {config.shape}")
    if policy.metadata.contract_json:
        expected_contract = json.loads(policy.metadata.contract_json)
        if expected_contract != config.metadata():
            raise SystemExit("policy and capture vision contracts differ")
    return policy


def infer_policy(policy, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    import torch

    tensor = torch.from_numpy(observation).unsqueeze(0)
    start = perf_counter()
    with torch.no_grad():
        normalized = policy(tensor)[0].numpy()
    inference_ms = (perf_counter() - start) * 1000.0
    scale = np.asarray(
        [
            policy.metadata.velocity_scale_m_s,
            policy.metadata.velocity_scale_m_s,
            policy.metadata.yawrate_scale_deg_s,
        ],
        dtype=np.float32,
    )
    return normalized, normalized * scale, inference_ms


if __name__ == "__main__":
    main()
