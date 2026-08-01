from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from flightrl.hardware.aideck_stream import AiDeckFrame

from .contract import GroundingResult


@dataclass(frozen=True, slots=True)
class FastPolicyClockConfig:
    grounding_stale_s: float = 1.0

    def __post_init__(self) -> None:
        if self.grounding_stale_s <= 0.0:
            raise ValueError("grounding stale interval must be positive")


class FastSemanticPolicyClock:
    """Clock visual policy inference from raw frames, not detector completions."""

    def __init__(
        self,
        policy,
        prompt: str,
        config: FastPolicyClockConfig | None = None,
    ) -> None:
        self.policy = policy
        self.prompt = prompt
        self.config = config or FastPolicyClockConfig()
        self.last_frame_index = -1
        self.last_grounding_index = -1
        self.latest_grounding: GroundingResult | None = None
        self.raw_frames_processed = 0
        self.grounding_updates = 0

    def poll(
        self,
        pipeline,
        telemetry: dict[str, float],
    ) -> tuple[AiDeckFrame, dict[str, Any]] | None:
        frame = pipeline.latest_frame()
        if frame is None or frame.index == self.last_frame_index:
            return None
        processed = pipeline.latest()
        grounding_is_new = False
        if processed is not None:
            result = processed[1]
            if result.frame_index != self.last_grounding_index:
                self.latest_grounding = result
                self.last_grounding_index = result.frame_index
                self.grounding_updates += 1
                grounding_is_new = True
        grounding = self.latest_grounding
        age_s = (
            float("inf")
            if grounding is None
            else max(0.0, frame.host_time_s - grounding.frame_host_time_s)
        )
        detection = None
        if (
            grounding is not None
            and grounding.best is not None
            and age_s <= self.config.grounding_stale_s
        ):
            detection = asdict(grounding.best)
        prediction = self.policy.step(
            frame=frame.pixels,
            telemetry=telemetry,
            prompt=self.prompt,
            detection=detection,
            update_semantic_memory=grounding_is_new,
        )
        prediction.update(
            {
                "fast_frame_clock": True,
                "policy_frame_index": frame.index,
                "grounding_frame_index": (
                    None if grounding is None else grounding.frame_index
                ),
                "grounding_age_s": age_s,
            }
        )
        self.last_frame_index = frame.index
        self.raw_frames_processed += 1
        return frame, prediction
