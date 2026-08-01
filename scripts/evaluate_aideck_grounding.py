from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import time

import numpy as np
from PIL import Image

from flightrl.hardware.aideck_stream import AiDeckFrame
from flightrl.semantic import (
    ClipCropVerifier,
    ClipVerifierConfig,
    GroundingDinoConfig,
    GroundingDinoGrounder,
    SemanticRunWriter,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate text grounding on archived AI Deck frames"
    )
    parser.add_argument("input", help="PNG/JPEG image or directory")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="artifacts/semantic/offline")
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--verifier-minimum-probability", type=float, default=0.60)
    parser.add_argument("--verifier-minimum-margin", type=float, default=0.45)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--min-frame-width", type=int, default=128)
    parser.add_argument("--require-detection", action="store_true")
    args = parser.parse_args()

    paths = image_paths(Path(args.input), args.max_frames)
    verifier = ClipCropVerifier(
        ClipVerifierConfig(
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
    inference_ms: list[float] = []
    detected = 0
    widths: list[int] = []
    means: list[float] = []
    manifest = {
        "mode": "offline",
        "prompt": args.prompt,
        "model_id": args.model_id,
        "device": args.device,
        "threshold": args.threshold,
        "verifier_model_id": verifier.config.model_id,
        "verifier_minimum_probability": verifier.config.minimum_probability,
        "verifier_minimum_margin": verifier.config.minimum_margin,
        "distractor_labels": list(grounder.config.distractor_labels),
        "sources": [str(path) for path in paths],
    }
    with SemanticRunWriter(args.output, manifest=manifest) as writer:
        for index, path in enumerate(paths, start=1):
            pixels = np.asarray(Image.open(path).convert("L"))
            frame = AiDeckFrame(
                index,
                time(),
                pixels.shape[1],
                pixels.shape[0],
                1,
                1,
                pixels,
            )
            result = grounder.detect(
                pixels,
                args.prompt,
                frame_index=index,
                frame_host_time_s=frame.host_time_s,
            )
            writer.write(frame, result)
            inference_ms.append(result.inference_ms)
            widths.append(result.image_width)
            means.append(result.source_mean)
            detected += int(result.best is not None)

    report = {
        "frames": len(paths),
        "prompt": args.prompt,
        "model_id": args.model_id,
        "threshold": args.threshold,
        "verifier_model_id": verifier.config.model_id,
        "frames_with_detection": detected,
        "detection_rate": detected / len(paths),
        "inference_ms_median": median(inference_ms),
        "inference_ms_max": max(inference_ms),
        "minimum_width": min(widths),
        "mean_brightness": sum(means) / len(means),
        "semantic_input_ready": min(widths) >= args.min_frame_width and detected > 0,
    }
    report_path = Path(args.output) / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_detection and detected == 0:
        raise SystemExit("no target detections passed the configured threshold")


def image_paths(path: Path, limit: int) -> list[Path]:
    if limit <= 0:
        raise ValueError("--max-frames must be positive")
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    extensions = {".png", ".jpg", ".jpeg"}
    paths = sorted(item for item in path.iterdir() if item.suffix.lower() in extensions)
    if not paths:
        raise FileNotFoundError(f"no PNG/JPEG frames found in {path}")
    if len(paths) <= limit:
        return paths
    indices = np.linspace(0, len(paths) - 1, limit, dtype=int)
    return [paths[index] for index in indices]


if __name__ == "__main__":
    main()
