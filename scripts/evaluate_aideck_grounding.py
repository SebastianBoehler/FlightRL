from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
from time import time

import numpy as np
from PIL import Image

from flightrl.hardware.aideck_stream import AiDeckFrame
from flightrl.semantic.aideck_pair_gate import evaluate_paired_gray4
from flightrl.semantic import (
    ClipCropVerifier,
    ClipVerifierConfig,
    GroundingDinoConfig,
    GroundingDinoGrounder,
    SemanticRunWriter,
)


@dataclass(frozen=True, slots=True)
class ArchivedFrame:
    source: Path
    index: int
    host_time_s: float
    pixels: np.ndarray
    capture_metadata: dict[str, object]


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

    archived_frames = load_archived_frames(Path(args.input), args.max_frames)
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
    proposed = 0
    proposal_count = 0
    verified_count = 0
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
        "sources": [
            {"path": str(frame.source), "frame_index": frame.index}
            for frame in archived_frames
        ],
        "capture_metadata": archived_frames[0].capture_metadata,
    }
    with SemanticRunWriter(args.output, manifest=manifest) as writer:
        for frame_source in archived_frames:
            pixels = frame_source.pixels
            index = frame_source.index + 1
            frame = AiDeckFrame(
                index,
                frame_source.host_time_s,
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
            proposed += int(result.best_proposal is not None)
            proposal_count += len(result.proposed_detections)
            verified_count += len(result.detections)

    report = {
        "frames": len(archived_frames),
        "prompt": args.prompt,
        "model_id": args.model_id,
        "threshold": args.threshold,
        "verifier_model_id": verifier.config.model_id,
        "frames_with_detection": detected,
        "detection_rate": detected / len(archived_frames),
        "frames_with_proposal": proposed,
        "proposal_rate": proposed / len(archived_frames),
        "proposals_total": proposal_count,
        "verified_detections_total": verified_count,
        "verification_rejection_rate": (
            (proposal_count - verified_count) / proposal_count
            if proposal_count
            else 0.0
        ),
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


def load_archived_frames(path: Path, limit: int) -> list[ArchivedFrame]:
    if path.is_file() and path.suffix.lower() == ".npz":
        return _load_npz_frames(path, limit)
    return [
        ArchivedFrame(
            source=image_path,
            index=index,
            host_time_s=time(),
            pixels=np.asarray(Image.open(image_path).convert("L")),
            capture_metadata={},
        )
        for index, image_path in enumerate(image_paths(path, limit))
    ]


def _load_npz_frames(path: Path, limit: int) -> list[ArchivedFrame]:
    if limit <= 0:
        raise ValueError("--max-frames must be positive")
    with np.load(path, allow_pickle=False) as artifact:
        required = {"decoded_frames", "host_time_s", "metadata_json"}
        if not required.issubset(artifact.files):
            raise ValueError(
                "AI Deck NPZ must contain decoded_frames, host_time_s, and metadata_json"
            )
        frames = np.asarray(artifact["decoded_frames"])
        host_times = np.asarray(artifact["host_time_s"], dtype=np.float64)
        metadata = json.loads(str(artifact["metadata_json"]))
    if frames.ndim != 3 or frames.dtype != np.uint8 or len(frames) == 0:
        raise ValueError("AI Deck decoded_frames must be non-empty [frames, height, width] uint8")
    if not isinstance(metadata, dict):
        raise ValueError("AI Deck metadata_json must decode to an object")
    if (
        host_times.shape != (len(frames),)
        or not np.isfinite(host_times).all()
        or np.any(np.diff(host_times) < 0.0)
    ):
        raise ValueError("AI Deck host_time_s must be finite and nondecreasing")
    count = min(limit, len(frames))
    indices = np.linspace(0, len(frames) - 1, count, dtype=int)
    return [
        ArchivedFrame(
            path,
            int(index),
            float(host_times[index]),
            frames[index].copy(),
            metadata,
        )
        for index in indices
    ]


def evaluate_paired_captures(
    positive_path: Path,
    negative_path: Path,
    *,
    sample_count: int,
) -> dict[str, object]:
    positive = load_archived_frames(positive_path, sample_count)
    negative = load_archived_frames(negative_path, sample_count)
    return evaluate_paired_gray4(
        np.stack([frame.pixels for frame in positive]),
        np.stack([frame.pixels for frame in negative]),
        positive_indices=[frame.index for frame in positive],
        negative_indices=[frame.index for frame in negative],
        positive_source=positive_path,
        negative_source=negative_path,
        positive_metadata=positive[0].capture_metadata,
        negative_metadata=negative[0].capture_metadata,
    )


if __name__ == "__main__":
    main()
