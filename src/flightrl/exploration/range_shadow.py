from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from math import isfinite, pi
from pathlib import Path

import numpy as np
import torch

from .range_checkpoint import load_range_checkpoint
from .range_mapper import RangeOccupancyMap, RangePose
from .range_observation import build_range_exploration_observation
from .range_safety import RangeClearanceHold, shield_range_exploration_action


RANGE_SHADOW_SCHEMA = "flightrl.range_exploration.replay_shadow.v2"
_HEADER = (
    "host_time_s",
    "crazyflie_time_ms",
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stateEstimate.yaw",
    "pm.vbat",
    "pm.state",
    "stateEstimate.roll",
    "stateEstimate.pitch",
    "stabilizer.roll",
    "stabilizer.pitch",
    "stabilizer.yaw",
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
    "range.zrange",
    "motion.motion",
    "motion.squal",
)
_AUTHORITY = {
    "training": False,
    "shadow": False,
    "deployment": False,
    "flight": False,
}


def range_replay_live_shadow_eligible(
    *,
    simulation_gate_passed: bool,
    replay_passed: bool,
) -> bool:
    if type(simulation_gate_passed) is not bool or type(replay_passed) is not bool:
        raise ValueError("range replay eligibility inputs must be bool")
    return simulation_gate_passed and replay_passed


def replay_range_shadow(
    checkpoint_path: str | Path,
    telemetry_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    checkpoint = Path(checkpoint_path)
    telemetry = Path(telemetry_path)
    model, evaluation = load_range_checkpoint(checkpoint)
    rows = _read_rows(telemetry)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    mapper = RangeOccupancyMap()
    previous_action = np.zeros(2, dtype=np.float32)
    previous_timestamp: int | None = None
    clearance_hold = RangeClearanceHold()
    reasons: Counter[str] = Counter()
    active_rows = 0
    overrides = 0
    records = []
    for row in rows:
        timestamp = _integer(row, "crazyflie_time_ms")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("range replay device timestamps must be strictly ordered")
        gap_ms = None if previous_timestamp is None else timestamp - previous_timestamp
        previous_timestamp = timestamp
        active = _number(row, "stateEstimate.z") >= 0.20
        raw_action = np.zeros(2, dtype=np.float32)
        shielded = np.zeros(2, dtype=np.float32)
        row_reasons: list[str] = []
        frontier_count = 0
        if active:
            active_rows += 1
            pose = RangePose(
                _number(row, "stateEstimate.x"),
                _number(row, "stateEstimate.y"),
                _number(row, "stabilizer.yaw") * pi / 180.0,
            )
            ranges, validity, range_reasons = _horizontal_ranges(row)
            row_reasons.extend(range_reasons)
            mapper.update(pose, ranges, validity)
            frontier_count = len(mapper.frontier_cells(pose))
            map_crop = mapper.exploration_crop(pose)
            observation = build_range_exploration_observation(
                map_crop,
                ranges / 4.0,
                validity,
                previous_action,
            )
            with torch.no_grad():
                action, _value = model.forward_step(
                    torch.from_numpy(observation[None, :])
                )
            raw_action = action[0].cpu().numpy().astype(np.float32)
            shielded, safety_reasons = _shield_action(
                raw_action, row, ranges, validity, map_crop, clearance_hold, gap_ms
            )
            row_reasons.extend(safety_reasons)
            if not np.array_equal(raw_action, shielded):
                overrides += 1
            previous_action = shielded.copy()
        for reason in set(row_reasons):
            reasons[reason] += 1
        records.append(
            {
                "crazyflie_time_ms": timestamp,
                "active": active,
                "frontier_count": frontier_count,
                "raw_policy_action": raw_action.tolist(),
                "shielded_policy_action": shielded.tolist(),
                "executed_action": [0.0, 0.0],
                "safety_reasons": sorted(set(row_reasons)),
                "controls_drone": False,
            }
        )
    actions_path = output / "shadow_actions.jsonl"
    actions_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    replay_passed = active_rows > 0 and not reasons
    simulation_gate_passed = evaluation["simulation_gate_passed"]
    report = {
        "schema": RANGE_SHADOW_SCHEMA,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "telemetry": str(telemetry.resolve()),
        "telemetry_sha256": _sha256(telemetry),
        "rows": len(rows),
        "active_rows": active_rows,
        "safety_override_rows": overrides,
        "safety_reason_counts": dict(sorted(reasons.items())),
        "replay_passed": replay_passed,
        "simulation_gate_passed": simulation_gate_passed,
        "exact_actor_replay": False,
        "previous_action_source": "unavailable_zero_initialized_then_shadow_shielded",
        "eligible_for_live_shadow": range_replay_live_shadow_eligible(
            simulation_gate_passed=simulation_gate_passed,
            replay_passed=replay_passed,
        ),
        "controls_drone": False,
        "authority": dict(_AUTHORITY),
    }
    (output / "shadow_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _HEADER:
            raise ValueError("range replay telemetry header is incompatible")
        rows = list(reader)
    if not rows:
        raise ValueError("range replay telemetry is empty")
    for row in rows:
        for name in _HEADER:
            _number(row, name)
    return rows


def _horizontal_ranges(
    row: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ranges = np.empty(4, dtype=np.float32)
    validity = np.ones(4, dtype=np.float32)
    reasons = []
    for index, name in enumerate(
        ("range.front", "range.back", "range.left", "range.right")
    ):
        value_mm = _number(row, name)
        if value_mm >= 32766.0:
            ranges[index] = 4.0
            validity[index] = 0.0
        elif 30.0 <= value_mm <= 4000.0 and value_mm.is_integer():
            ranges[index] = value_mm / 1000.0
        else:
            ranges[index] = 4.0
            validity[index] = 0.0
            reasons.append("invalid_horizontal_range")
    return ranges, validity, reasons


def _shield_action(
    action: np.ndarray,
    row: dict[str, str],
    ranges: np.ndarray,
    validity: np.ndarray,
    map_crop: np.ndarray,
    clearance_hold: RangeClearanceHold,
    gap_ms: int | None,
) -> tuple[np.ndarray, list[str]]:
    result, _emergency, range_reasons = shield_range_exploration_action(
        action, ranges, validity, map_crop
    )
    result, range_reasons = clearance_hold.apply(result, range_reasons)
    reasons = []
    if gap_ms is not None and gap_ms > 250:
        reasons.append("stale_telemetry")
    if _number(row, "motion.squal") < 80.0:
        reasons.append("low_flow_quality")
    if _number(row, "pm.vbat") < 3.50 or int(_number(row, "pm.state")) in {3, 4}:
        reasons.append("power_state")
    if max(abs(_number(row, "stateEstimate.roll")), abs(_number(row, "stateEstimate.pitch"))) > 20.0:
        reasons.append("excessive_tilt")
    up_mm = _number(row, "range.up")
    if 30.0 <= up_mm < 200.0:
        reasons.append("up_clearance")
    reasons.extend(range_reasons)
    fatal = set(reasons) - {
        "forward_clearance_override",
        "horizontal_clearance_override",
        "estimated_map_clearance_override",
        "clearance_hold",
    }
    if fatal:
        result[:] = 0.0
    return result, reasons


def _number(row: dict[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"range replay {name} must be numeric") from exc
    if not isfinite(value):
        raise ValueError(f"range replay {name} must be finite")
    return value


def _integer(row: dict[str, str], name: str) -> int:
    value = _number(row, name)
    if not value.is_integer() or not 0.0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"range replay {name} must be uint32")
    return int(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
