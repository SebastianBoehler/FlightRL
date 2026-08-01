from __future__ import annotations

import numpy as np

from flightrl.puffer4_door_self_mask import apply_door_self_mask


def test_self_mask_neutralizes_guard_regions_without_erasing_center() -> None:
    frame = np.full((244, 324), 68, dtype=np.uint8)
    frame[:61] = 0
    masked = apply_door_self_mask(frame)
    fill = np.mean(frame).astype(np.uint8)

    assert masked.shape == frame.shape
    assert masked.dtype == frame.dtype
    assert masked[45, 40] == fill
    assert masked[45, 285] == fill
    assert masked[122, 162] == frame[122, 162]
    assert not np.shares_memory(masked, frame)
