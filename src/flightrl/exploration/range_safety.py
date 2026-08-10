from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RangeClearanceHold:
    remaining_clear_steps: int = 0

    def reset(self) -> None:
        self.remaining_clear_steps = 0

    def apply(
        self,
        action: np.ndarray,
        reasons: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        result = np.asarray(action, dtype=np.float32).copy()
        trigger = bool(
            set(reasons)
            & {
                "emergency_horizontal_clearance",
                "forward_clearance_override",
                "horizontal_clearance_override",
                "estimated_map_clearance_override",
            }
        )
        output_reasons = list(reasons)
        if trigger:
            self.remaining_clear_steps = 10
            result[0] = 0.0
        elif self.remaining_clear_steps > 0:
            result[0] = 0.0
            self.remaining_clear_steps -= 1
            output_reasons.append("clearance_hold")
        return result, output_reasons


def shield_range_exploration_action(
    action: np.ndarray,
    ranges_m: np.ndarray,
    validity: np.ndarray,
    map_crop: np.ndarray,
) -> tuple[np.ndarray, bool, list[str]]:
    command = np.asarray(action, dtype=np.float32)
    ranges = np.asarray(ranges_m, dtype=np.float32)
    valid = np.asarray(validity, dtype=np.float32)
    map_value = np.asarray(map_crop, dtype=np.float32)
    if (
        command.shape != (2,)
        or ranges.shape != (4,)
        or valid.shape != (4,)
        or map_value.shape != (4, 32, 32)
        or not np.isfinite(command).all()
        or not np.isfinite(ranges).all()
        or not np.isfinite(valid).all()
        or not np.isfinite(map_value).all()
    ):
        raise ValueError("range exploration safety inputs are incompatible")
    result = command.copy()
    reasons: list[str] = []
    finite = ranges[valid.astype(bool)]
    emergency = bool(len(finite) and float(np.min(finite)) < 0.20)
    if emergency:
        result[:] = 0.0
        reasons.append("emergency_horizontal_clearance")
        return result, True, reasons
    if bool(valid[0]) and float(ranges[0]) < 0.35:
        result[0] = 0.0
        reasons.append("forward_clearance_override")
    elif len(finite) and float(np.min(finite)) < 0.35:
        result[0] = 0.0
        reasons.append("horizontal_clearance_override")
    occupied = np.argwhere(map_value[2, :17] > 0.5)
    if len(occupied):
        offsets_m = (occupied.astype(np.float32) - 16.0) * 0.20
        if float(np.min(np.linalg.norm(offsets_m, axis=1))) <= 0.60:
            result[0] = 0.0
            reasons.append("estimated_map_clearance_override")
    return result, False, reasons
