from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw

from flightrl.hardware.aideck_stream import AiDeckFrame

from .contract import GroundingResult


class SemanticRunWriter:
    def __init__(self, output_dir: str | Path, *, manifest: Mapping[str, Any]) -> None:
        self.output_dir = Path(output_dir)
        self.frame_dir = self.output_dir / "frames"
        self.annotated_dir = self.output_dir / "annotated"
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "manifest.json").write_text(
            json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n"
        )
        self._events = (self.output_dir / "events.jsonl").open("w")
        self._written_frames: set[int] = set()

    def write(
        self,
        frame: AiDeckFrame,
        grounding: GroundingResult,
        *,
        telemetry: Mapping[str, Any] | None = None,
    ) -> Path:
        frame_path = self.frame_dir / f"frame-{frame.index:06d}.png"
        annotated_path = self.annotated_dir / f"frame-{frame.index:06d}.png"
        if frame.index not in self._written_frames:
            image = Image.fromarray(frame.pixels)
            image.save(frame_path)
            annotate_grounding(image, grounding).save(annotated_path)
            self._written_frames.add(frame.index)
        event = {
            "frame_path": str(frame_path),
            "annotated_path": str(annotated_path),
            "grounding": grounding.to_dict(),
            "telemetry": dict(telemetry or {}),
            "controls_drone": False,
        }
        self._events.write(json.dumps(event, sort_keys=True) + "\n")
        self._events.flush()
        return annotated_path

    def close(self) -> None:
        self._events.close()

    def __enter__(self) -> "SemanticRunWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def annotate_grounding(image: Image.Image, grounding: GroundingResult) -> Image.Image:
    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    for detection in grounding.detections:
        box = detection.box
        xy = (
            round(box.x_min * width),
            round(box.y_min * height),
            round(box.x_max * width),
            round(box.y_max * height),
        )
        draw.rectangle(xy, outline=(255, 64, 64), width=max(1, width // 128))
        draw.text((xy[0], max(0, xy[1] - 10)), f"{detection.label} {detection.confidence:.2f}")
    return annotated
