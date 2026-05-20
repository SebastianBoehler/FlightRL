from __future__ import annotations

import numpy as np


def trajectory_quality(trajectory, pose_xyz: np.ndarray, duration_s: float, path_length_m: float) -> dict[str, float]:
    if len(trajectory) == 0:
        return empty_trajectory_quality()
    straight = float(np.linalg.norm(pose_xyz[-1] - pose_xyz[0])) if len(pose_xyz) >= 2 else 0.0
    step_dist = np.linalg.norm(np.diff(pose_xyz, axis=0), axis=1) if len(pose_xyz) >= 2 else np.asarray([], dtype=np.float32)
    times = np.asarray([pose.time_s for pose in trajectory], dtype=np.float32)
    dt = np.diff(times)
    valid_dt = dt > 1e-6
    speeds = step_dist[valid_dt] / dt[valid_dt] if len(step_dist) and np.any(valid_dt) else np.asarray([], dtype=np.float32)
    yaws = np.unwrap(np.deg2rad([pose.yaw_deg for pose in trajectory]))
    return {
        "straight_line_m": straight,
        "loop_closure_m": straight,
        "mean_speed_m_s": float(np.mean(speeds)) if len(speeds) else 0.0,
        "p95_speed_m_s": float(np.quantile(speeds, 0.95)) if len(speeds) else 0.0,
        "max_step_speed_m_s": float(np.max(speeds)) if len(speeds) else 0.0,
        "z_std_m": float(np.std(pose_xyz[:, 2])) if len(pose_xyz) else 0.0,
        "yaw_span_deg": float(np.rad2deg(np.ptp(yaws))) if len(yaws) else 0.0,
        "path_efficiency": straight / max(path_length_m, 1e-6),
        "mean_sample_rate_hz": float((len(trajectory) - 1) / max(duration_s, 1e-6)) if len(trajectory) > 1 else 0.0,
    }


def empty_trajectory_quality() -> dict[str, float]:
    return {
        "straight_line_m": 0.0,
        "loop_closure_m": 0.0,
        "mean_speed_m_s": 0.0,
        "p95_speed_m_s": 0.0,
        "max_step_speed_m_s": 0.0,
        "z_std_m": 0.0,
        "yaw_span_deg": 0.0,
        "path_efficiency": 0.0,
        "mean_sample_rate_hz": 0.0,
    }
