from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

from flightrl.hardware.avoidance_policy import AvoidanceCommand, RangerReading, normalize_reading
from flightrl.hardware.avoidance_ttc import min_horizontal_ttc_s


TTC_OBSERVATION_SIZE = 14


class TTCAvoidancePolicy(nn.Module):
    def __init__(self, hidden_size: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TTC_OBSERVATION_SIZE, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 4),
        )

    def forward(self, observations):
        tensor = torch.as_tensor(observations, dtype=torch.float32)
        return self.net(tensor)


def ttc_observation(
    reading: RangerReading,
    range_rate_m_s: RangerReading,
    *,
    max_rate_m_s: float = 4.0,
    ttc_horizon_s: float = 0.7,
) -> np.ndarray:
    rates = np.asarray(
        [
            range_rate_m_s.front_m,
            range_rate_m_s.back_m,
            range_rate_m_s.left_m,
            range_rate_m_s.right_m,
            range_rate_m_s.up_m,
            range_rate_m_s.zrange_m,
        ],
        dtype=np.float32,
    )
    horizontal_min_range = min(reading.front_m, reading.back_m, reading.left_m, reading.right_m)
    return np.concatenate(
        [
            normalize_reading(reading),
            np.clip(rates / max(max_rate_m_s, 1e-6), -1.0, 1.0),
            np.asarray(
                [
                    np.clip(horizontal_min_range / 4.0, 0.0, 1.0),
                    ttc_urgency(min_horizontal_ttc_s(reading, range_rate_m_s), ttc_horizon_s),
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


def rate_from_telemetry(values: Mapping[str, float]) -> RangerReading:
    return RangerReading(
        front_m=_float(values, "range_rate_front_m_s"),
        back_m=_float(values, "range_rate_back_m_s"),
        left_m=_float(values, "range_rate_left_m_s"),
        right_m=_float(values, "range_rate_right_m_s"),
        up_m=_float(values, "range_rate_up_m_s"),
        zrange_m=_float(values, "range_rate_zrange_m_s"),
    )


def command_from_ttc_model(
    model: TTCAvoidancePolicy,
    reading: RangerReading,
    range_rate_m_s: RangerReading,
    *,
    max_speed_m_s: float = 0.65,
    max_yawrate_deg_s: float = 45.0,
) -> AvoidanceCommand:
    observation = ttc_observation(reading, range_rate_m_s)[None, :]
    with torch.no_grad():
        raw = model(observation).squeeze(0).cpu().numpy()
    vx_m_s, vy_m_s = _clip_norm(float(raw[0]), float(raw[1]), max_speed_m_s)
    return AvoidanceCommand(vx_m_s, vy_m_s, float(raw[2]), float(raw[3])).clipped(
        max_speed=max_speed_m_s,
        max_yawrate=max_yawrate_deg_s,
    )


def load_ttc_policy(path: str | Path) -> TTCAvoidancePolicy:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["state_dict"]
    hidden_size = int(checkpoint.get("hidden_size") or state["net.0.weight"].shape[0])
    model = TTCAvoidancePolicy(hidden_size=hidden_size)
    model.load_state_dict(state)
    model.eval()
    return model


def ttc_urgency(ttc_s: float, horizon_s: float) -> float:
    if not np.isfinite(ttc_s) or ttc_s >= horizon_s:
        return 0.0
    return float(np.clip((horizon_s - max(ttc_s, 0.0)) / max(horizon_s, 1e-6), 0.0, 1.0))


def _float(values: Mapping[str, float], key: str) -> float:
    try:
        return float(values.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _clip_norm(vx_m_s: float, vy_m_s: float, max_speed_m_s: float) -> tuple[float, float]:
    vector = np.asarray([vx_m_s, vy_m_s], dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= max_speed_m_s or norm <= 1e-9:
        return float(vector[0]), float(vector[1])
    scaled = vector * (max_speed_m_s / norm)
    return float(scaled[0]), float(scaled[1])
