from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from flightrl.hardware.aideck_stream import AiDeckUdpStream
from flightrl.semantic import (
    AsyncGroundingPipeline,
    ClipCropVerifier,
    ClipVerifierConfig,
    GroundingDinoConfig,
    GroundingDinoGrounder,
    SemanticRunWriter,
    collect_camera_only,
    require_semantic_frame,
    write_capture_summary,
)
def main() -> None:
    args = parse_args()
    if args.dry_run:
        print_plan(args)
        return

    output = Path(args.output or default_output(args.prompt))
    verifier = ClipCropVerifier(
        ClipVerifierConfig(
            model_id=args.verifier_model_id,
            device=args.device,
            minimum_probability=args.verifier_minimum_probability,
            minimum_margin=args.verifier_minimum_margin,
        )
    )
    grounder = GroundingDinoGrounder(
        GroundingDinoConfig(
            model_id=args.model_id,
            device=args.device,
            threshold=args.threshold,
        ),
        verifier=verifier,
    )
    stream = AiDeckUdpStream(
        host=args.aideck_host,
        port=args.aideck_port,
        bind_host=args.bind_host,
        bind_port=args.bind_port,
        timeout_s=args.camera_timeout_s,
    )
    pipeline = AsyncGroundingPipeline(stream, grounder, args.prompt)
    manifest = {
        "prompt": args.prompt,
        "model_id": args.model_id,
        "device": args.device,
        "threshold": args.threshold,
        "distractor_labels": list(grounder.config.distractor_labels),
        "verifier_model_id": args.verifier_model_id,
        "verifier_minimum_probability": args.verifier_minimum_probability,
        "verifier_minimum_margin": args.verifier_minimum_margin,
        "controls_drone": False,
        "authority": "none",
    }
    try:
        pipeline.start()
        frame = pipeline.wait_for_frame(args.camera_timeout_s)
        require_semantic_frame(
            frame,
            min_width=args.min_frame_width,
            min_mean=args.min_frame_mean,
        )
        pipeline.wait_for_result(args.first_result_timeout_s)
        with SemanticRunWriter(output, manifest=manifest) as writer:
            summary = collect_camera_only(
                pipeline,
                writer,
                duration_s=args.duration_s,
            )
        summary.update(
            {
                "output": str(output),
                "camera_width": frame.width,
                "camera_height": frame.height,
                "camera_mean": float(frame.pixels.mean()),
            }
        )
        print(
            "semantic capture complete: "
            f"{write_capture_summary(output, summary)}"
        )
    finally:
        pipeline.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture non-actuating text-grounding data"
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output")
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument(
        "--verifier-model-id",
        default="openai/clip-vit-base-patch32",
    )
    parser.add_argument("--verifier-minimum-probability", type=float, default=0.60)
    parser.add_argument("--verifier-minimum-margin", type=float, default=0.45)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--min-frame-width", type=int, default=128)
    parser.add_argument("--min-frame-mean", type=float, default=8.0)
    parser.add_argument("--aideck-host", default="192.168.4.1")
    parser.add_argument("--aideck-port", type=int, default=5000)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--bind-port", type=int, default=5001)
    parser.add_argument("--camera-timeout-s", type=float, default=10.0)
    parser.add_argument("--first-result-timeout-s", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def print_plan(args: argparse.Namespace) -> None:
    print(
        f"dry_run prompt={args.prompt!r} mode=camera_only "
        f"controls_drone=false duration_s={args.duration_s:.1f}"
    )


def default_output(prompt: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "".join(character if character.isalnum() else "-" for character in prompt)
    return f"artifacts/semantic/{stamp}-{slug.strip('-').lower()}"


if __name__ == "__main__":
    main()
