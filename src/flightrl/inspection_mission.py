"""Pixel-only marker memory; independent truth evaluation never feeds memory."""

from dataclasses import dataclass, field

import numpy as np

from flightrl.sixdof.geometry import quat_to_matrix

QUALITY = {
    "min_pixels": 64,
    "min_visible_fraction": 0.95,
    "max_distance_m": 3.0,
    "min_facing_cosine": 0.75,
    "blur": "instantaneous_ideal_camera_no_motion_blur_model",
}


def detect_markers(frame):
    """Diagnostic red/green/blue appearance identities, not evaluator IDs.

    Deliberately assumes unique solid colored square markers and neutral walls.
    No scene, pose, panel table, segmentation mask or target count is accepted.
    """
    if frame.shape != (48, 64, 3) or frame.dtype != np.uint8:
        raise ValueError("expected RGB uint8 camera frame")
    found = []
    for channel, name in enumerate(("red", "green", "blue")):
        other = np.delete(frame, channel, axis=2).max(axis=2)
        mask = (frame[:, :, channel] >= 180) & (other <= 60)
        y, x = np.nonzero(mask)
        if len(x) < 4:
            continue
        width, height = int(x.max() - x.min() + 1), int(y.max() - y.min() + 1)
        fill = len(x) / (width * height)
        interior = x.min() > 0 and x.max() < 63 and y.min() > 0 and y.max() < 47
        useful = (
            len(x) >= 64 and fill >= 0.95 and 0.65 <= width / height <= 1.5 and interior
        )
        found.append(
            {
                "marker": name,
                "pixels": len(x),
                "bbox_xyxy": [int(x.min()), int(y.min()), int(x.max()), int(y.max())],
                "useful_view_observed": bool(useful),
            }
        )
    return found


@dataclass
class InspectionMemory:
    budget_ticks: int
    discovered: set = field(default_factory=set)
    inspected: set = field(default_factory=set)
    events: list = field(default_factory=list)
    next_tick: int = 0
    duplicate_views: int = 0
    status: str = "running"

    def __post_init__(self):
        if type(self.budget_ticks) is not int or self.budget_ticks < 1:
            raise ValueError("positive integer budget required")

    def observe(self, frame, tick):
        if self.status != "running" or tick != self.next_tick:
            raise ValueError(
                "mission observations must be sequential and within budget"
            )
        for detection in detect_markers(frame):
            key = detection["marker"]
            if key not in self.discovered:
                self.discovered.add(key)
                self.events.append({"tick": tick, "type": "discovered", "marker": key})
            if detection["useful_view_observed"]:
                if key in self.inspected:
                    self.duplicate_views += 1
                else:
                    self.inspected.add(key)
                    self.events.append(
                        {"tick": tick, "type": "inspected_observed", "marker": key}
                    )
        self.next_tick += 1
        if self.next_tick == self.budget_ticks:
            self.status = "budget_exhausted_coverage_unknown"
            self.events.append({"tick": tick, "type": "budget_exhausted"})


def evaluate_views(scene, positions, quaternions, counts):
    """Evaluator only: ideal-image useful view, not learned image-quality proof."""
    r = quat_to_matrix(quaternions)
    origin = positions + np.einsum(
        "nij,j->ni", r, np.array([0.035, 0, 0.012], np.float32)
    )
    delta = origin[:, None, :] - scene.panels[None, :, :3]
    distance = np.linalg.norm(delta, axis=2)
    normal = np.cross(scene.panels[:, 3:6], scene.panels[:, 6:9])
    cosine = np.einsum("npj,pj->np", delta, normal) / np.maximum(distance, 1e-8)
    visible = counts[:, :, 0]
    projected = counts[:, :, 1]
    # Reject clipped panels: visible/projected counts alone miss image-boundary clipping.
    contained = np.ones_like(visible, dtype=bool)
    fy = np.tan(1.099557429 / 2)
    for su, sv in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        corners = (
            scene.panels[:, :3]
            + su * scene.panels[:, 3:6] * scene.panels[:, 9:10]
            + sv * scene.panels[:, 6:9] * scene.panels[:, 10:11]
        )
        body = np.einsum("nji,npj->npi", r, corners[None, :, :] - origin[:, None, :])
        contained &= (
            (body[:, :, 0] > 0)
            & (abs(body[:, :, 1]) < body[:, :, 0] * fy * 64 / 48)
            & (abs(body[:, :, 2]) < body[:, :, 0] * fy)
        )
    return (
        contained
        & (visible >= QUALITY["min_pixels"])
        & (visible >= 0.95 * projected)
        & (distance <= 3)
        & (cosine >= 0.75)
    )
