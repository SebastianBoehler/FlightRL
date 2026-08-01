from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from flightrl.semantic import (
    ClipCropVerifier,
    ClipVerifierConfig,
    GroundingDinoConfig,
    GroundingDinoGrounder,
)
from flightrl.semantic.resolution_sweep import (
    ResolutionVariant,
    evaluate_variant,
    variant_metrics,
)


DEFAULT_RESOLUTIONS = (
    (324, 244),
    (243, 183),
    (162, 122),
    (128, 96),
    (96, 72),
    (64, 48),
    (48, 36),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure semantic grounding degradation versus resolution"
    )
    parser.add_argument("input", help="Directory of QVGA reference frames")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="artifacts/semantic/resolution-sweep")
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--verifier-minimum-probability", type=float, default=0.60)
    parser.add_argument("--verifier-minimum-margin", type=float, default=0.45)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--bits", default="8,4")
    args = parser.parse_args()

    paths = image_paths(Path(args.input), args.max_frames)
    bits = parse_bits(args.bits)
    requested_variants = [
        ResolutionVariant(width, height, depth)
        for width, height in DEFAULT_RESOLUTIONS
        for depth in bits
    ]
    baseline_variant = ResolutionVariant(324, 244, 8)
    variants = [
        baseline_variant,
        *(variant for variant in requested_variants if variant != baseline_variant),
    ]
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
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    baseline = None
    rows: list[dict] = []
    for variant in variants:
        evaluated = evaluate_variant(
            paths,
            prompt=args.prompt,
            grounder=grounder,
            variant=variant,
            output_dir=output,
        )
        if baseline is None:
            baseline = evaluated
        metrics = variant_metrics(evaluated, baseline)
        rows.append(
            {
                "variant": variant.name,
                "width": variant.width,
                "height": variant.height,
                "bits": variant.bits,
                **metrics,
            }
        )
        print(
            f"{variant.name}: detection={metrics['detection_rate']:.2f} "
            f"recall={metrics['baseline_recall']:.2f} "
            f"iou={metrics['median_box_iou']:.2f} "
            f"retained={metrics['signal_retained']}"
        )
    report = {
        "prompt": args.prompt,
        "model_id": args.model_id,
        "threshold": args.threshold,
        "distractor_labels": list(grounder.config.distractor_labels),
        "verifier_model_id": verifier.config.model_id,
        "verifier_minimum_probability": verifier.config.minimum_probability,
        "verifier_minimum_margin": verifier.config.minimum_margin,
        "source_frames": [str(path) for path in paths],
        "baseline": baseline_variant.name,
        "variants": rows,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    write_csv(output / "report.csv", rows)


def image_paths(path: Path, limit: int) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    if limit <= 0:
        raise ValueError("--max-frames must be positive")
    paths = sorted(
        item
        for item in path.iterdir()
        if item.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not paths:
        raise FileNotFoundError(f"no images found in {path}")
    if len(paths) <= limit:
        return paths
    step = (len(paths) - 1) / (limit - 1) if limit > 1 else 0.0
    return [paths[round(index * step)] for index in range(limit)]


def parse_bits(value: str) -> tuple[int, ...]:
    bits = tuple(dict.fromkeys(int(item) for item in value.split(",")))
    if not bits or any(item not in {4, 8} for item in bits):
        raise ValueError("--bits must contain 4 and/or 8")
    return bits


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
