from __future__ import annotations

from dataclasses import asdict, dataclass

from flightrl.replay import aligned_compare

from .command_replay import replay_velocity_commands
from .geometry import BoxRoom


DEFAULT_SIGNALS = (
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stabilizer.yaw",
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
)


@dataclass(frozen=True, slots=True)
class ReplayCandidate:
    velocity_gain: float
    yawrate_scale: float
    max_dt_s: float
    hold_z_m: float | None
    override_z_m: float | None
    command_frame: str = "body"
    yaw_source: str = "logged"
    vx_sign: float = 1.0
    vy_sign: float = 1.0


def sweep_command_replay(
    rows: list[dict[str, str]],
    *,
    room: BoxRoom | None,
    candidates: list[ReplayCandidate],
    signals: tuple[str, ...] = DEFAULT_SIGNALS,
) -> list[dict]:
    records = []
    for candidate in candidates:
        sim_rows, real_rows = replay_velocity_commands(
            rows,
            room=room,
            override_z_m=candidate.override_z_m,
            hold_z_m=candidate.hold_z_m,
            velocity_gain=candidate.velocity_gain,
            yawrate_scale=candidate.yawrate_scale,
            command_frame=candidate.command_frame,
            yaw_source=candidate.yaw_source,
            vx_sign=candidate.vx_sign,
            vy_sign=candidate.vy_sign,
            max_dt_s=candidate.max_dt_s,
        )
        comparison = aligned_compare(real_rows, sim_rows, list(signals))
        metrics = score_comparison(comparison)
        records.append({"params": asdict(candidate), "score": metrics["score"], "metrics": metrics})
    return sorted(records, key=lambda item: item["score"])


def candidate_grid(
    *,
    velocity_gains: list[float],
    yawrate_scales: list[float],
    max_dt_values: list[float],
    override_z_m: float | None,
    hold_z_values: list[float | None],
    command_frames: list[str] | None = None,
    yaw_sources: list[str] | None = None,
    vx_signs: list[float] | None = None,
    vy_signs: list[float] | None = None,
) -> list[ReplayCandidate]:
    command_frames = command_frames or ["body"]
    yaw_sources = yaw_sources or ["logged"]
    vx_signs = vx_signs or [1.0]
    vy_signs = vy_signs or [1.0]
    return [
        ReplayCandidate(v_gain, yaw_scale, max_dt, hold_z, override_z_m, frame, yaw_source, vx_sign, vy_sign)
        for v_gain in velocity_gains
        for yaw_scale in yawrate_scales
        for max_dt in max_dt_values
        for hold_z in hold_z_values
        for frame in command_frames
        for yaw_source in yaw_sources
        for vx_sign in vx_signs
        for vy_sign in vy_signs
    ]


def score_comparison(comparison: dict) -> dict[str, float]:
    signals = comparison.get("signals", {})
    state_xy = worst_rmse(signals, ("stateEstimate.x", "stateEstimate.y"))
    state_z = worst_rmse(signals, ("stateEstimate.z",))
    yaw = worst_rmse(signals, ("stabilizer.yaw",))
    ranges = worst_rmse(signals, ("range.",))
    xy_yaw_score = state_xy + 0.01 * yaw
    state_bridge_score = state_xy + 0.25 * state_z + 0.01 * yaw
    score = state_xy + 0.25 * state_z + 0.002 * ranges + 0.01 * yaw
    return {
        "score": score,
        "xy_yaw_score": xy_yaw_score,
        "state_bridge_score": state_bridge_score,
        "samples": float(comparison.get("samples", 0)),
        "overlap_duration_s": float(comparison.get("overlap_duration_s", 0.0)),
        "worst_xy_state_rmse_m": state_xy,
        "z_rmse_m": state_z,
        "yaw_rmse_deg": yaw,
        "worst_range_rmse_mm": ranges,
    }


def worst_rmse(signals: dict, keys_or_prefixes: tuple[str, ...]) -> float:
    values = []
    for key, metrics in signals.items():
        if any(key == item or key.startswith(item) for item in keys_or_prefixes) and "rmse" in metrics:
            values.append(float(metrics["rmse"]))
    return max(values) if values else float("inf")
