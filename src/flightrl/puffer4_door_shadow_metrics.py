from __future__ import annotations

import json

import numpy as np

from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
)

MIN_SHADOW_FRAME_WIDTH = 128
MIN_SHADOW_FRAME_HEIGHT = 96
MIN_SEARCH_PHASE_ROWS = 5
MIN_TARGET_PHASE_ROWS = 5
MIN_SAMPLED_COVERAGE_S = 20.0
MAX_FRAME_ASPECT_ERROR = 0.02


def detection_yaw_alignment(
    rows: list[dict],
    *,
    center_deadband: float = 0.08,
    yaw_field: str = "action_yaw",
) -> tuple[int, float | None]:
    aligned: list[bool] = []
    for row in rows:
        encoded = row.get("detection")
        if not encoded:
            continue
        detection = json.loads(encoded)
        box = detection["box"]
        horizontal_error = (
            0.5 * (float(box["x_min"]) + float(box["x_max"])) - 0.5
        )
        if abs(horizontal_error) < center_deadband:
            continue
        aligned.append(float(row[yaw_field]) * horizontal_error < 0.0)
    return len(aligned), None if not aligned else sum(aligned) / len(aligned)


def shadow_capture_contract(
    rows: list[dict],
    phases: dict[str, int],
) -> dict:
    required = ("frame_host_time_s", "frame_width", "frame_height")
    if not rows or any(
        any(row.get(field) is None for field in required) for row in rows
    ):
        return {
            "sampled_coverage_s": None,
            "timestamps_strictly_increasing": False,
            "frame_width": None,
            "frame_height": None,
            "frame_geometry_contract_passed": False,
            "phase_coverage_passed": False,
        }
    timestamps = np.asarray(
        [float(row["frame_host_time_s"]) for row in rows],
        dtype=np.float64,
    )
    intervals = np.diff(timestamps)
    increasing = bool(len(intervals) and np.all(intervals > 0.0))
    sampled_coverage = (
        float(timestamps[-1] - timestamps[0] + np.median(intervals))
        if increasing
        else None
    )
    widths = {int(row["frame_width"]) for row in rows}
    heights = {int(row["frame_height"]) for row in rows}
    width = next(iter(widths)) if len(widths) == 1 else None
    height = next(iter(heights)) if len(heights) == 1 else None
    geometry_passed = bool(
        width is not None
        and height is not None
        and width >= MIN_SHADOW_FRAME_WIDTH
        and height >= MIN_SHADOW_FRAME_HEIGHT
        and abs(width / height - 4.0 / 3.0) <= MAX_FRAME_ASPECT_ERROR
    )
    target_phase_rows = sum(
        phases[phase] for phase in ("track", "approach", "recover")
    )
    return {
        "sampled_coverage_s": sampled_coverage,
        "timestamps_strictly_increasing": increasing,
        "frame_width": width,
        "frame_height": height,
        "frame_geometry_contract_passed": geometry_passed,
        "phase_coverage_passed": bool(
            phases["search"] >= MIN_SEARCH_PHASE_ROWS
            and target_phase_rows >= MIN_TARGET_PHASE_ROWS
        ),
    }


def shadow_stream_metrics(rows: list[dict]) -> dict:
    frame_indices = np.asarray(
        [int(row["frame_index"]) for row in rows],
        dtype=np.int64,
    )
    frame_deltas = np.diff(frame_indices)
    frame_ordered = bool(
        len(frame_deltas) == 0 or np.all(frame_deltas > 0)
    )
    frame_gaps = int(
        np.maximum(frame_deltas - 1, 0).sum()
        if frame_ordered
        else 0
    )
    dropped_values = {
        int(row["stream_dropped_frames"]) for row in rows
    }
    grounding_indices = np.asarray(
        [int(row["grounding_result_frame_index"]) for row in rows],
        dtype=np.int64,
    )
    grounding_ordered = bool(
        np.all(grounding_indices <= frame_indices)
        and (
            len(grounding_indices) <= 1
            or np.all(np.diff(grounding_indices) >= 0)
        )
    )
    update_rows = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(np.diff(grounding_indices) != 0) + 1,
        )
    )
    times = np.asarray(
        [float(row["frame_host_time_s"]) for row in rows],
        dtype=np.float64,
    )
    update_times = times[update_rows]
    update_intervals = np.diff(update_times)
    span_s = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    result_rate_hz = (
        float((len(update_rows) - 1) / span_s)
        if span_s > 0.0
        else 0.0
    )
    inference = np.asarray(
        [float(row["grounding_inference_ms"]) for row in rows],
        dtype=np.float64,
    )
    ages = np.asarray(
        [
            float(row["grounding_age_s"])
            for row in rows
            if row.get("grounding_age_s") is not None
        ],
        dtype=np.float64,
    )
    age_p95 = (
        None if not len(ages) else float(np.percentile(ages, 95))
    )
    return {
        "frame_indices_strictly_increasing": frame_ordered,
        "frame_index_gap_count": frame_gaps,
        "stream_dropped_frames": (
            next(iter(dropped_values))
            if len(dropped_values) == 1
            else None
        ),
        "stream_drop_counter_consistent": len(dropped_values) == 1,
        "grounding_result_frame_order_passed": grounding_ordered,
        "grounding_unique_results": len(update_rows),
        "grounding_result_rate_hz": result_rate_hz,
        "grounding_update_interval_s_p95": (
            None
            if not len(update_intervals)
            else float(np.percentile(update_intervals, 95))
        ),
        "grounding_inference_ms_p95": float(
            np.percentile(inference, 95)
        ),
        "grounding_age_margin_s_p05": (
            None
            if age_p95 is None
            else (
                FIXED_DOOR_EVIDENCE_AGE_CONTRACT.maximum_evidence_age_s
                - age_p95
            )
        ),
    }
