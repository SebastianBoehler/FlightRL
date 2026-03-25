from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .config import FlightConfig


if TYPE_CHECKING:
    from .renderer import DroneFrame


FORCE_STYLES = {
    "thrust": {"color": "#2f7f7a", "label": "Thrust", "scale": 0.08, "max_norm": 1.2},
    "drag": {"color": "#7d8597", "label": "Drag", "scale": 0.28, "max_norm": 1.0},
    "gravity": {"color": "#b26a4f", "label": "Gravity", "scale": 0.08, "max_norm": 1.0},
    "net": {"color": "#6d597a", "label": "Net", "scale": 0.12, "max_norm": 1.1},
    "wind": {"color": "#4c956c", "label": "Wind", "scale": 0.36, "max_norm": 1.0},
}


def compute_force_vectors(config: FlightConfig, frame: DroneFrame) -> dict[str, np.ndarray]:
    mass = config.drone.mass
    thrust_total = float(sum(frame.motor_thrusts))
    rel_vx = frame.vx - frame.wind_x
    rel_vz = frame.vz - frame.wind_z
    thrust = np.array(
        [-thrust_total * np.sin(frame.pitch), thrust_total * np.cos(frame.pitch)],
        dtype=np.float32,
    )
    drag = np.array([-config.drone.drag * rel_vx, -config.drone.drag * rel_vz], dtype=np.float32)
    gravity = np.array([0.0, -mass * config.drone.gravity], dtype=np.float32)
    net = mass * np.array([frame.ax, frame.az], dtype=np.float32)
    wind = np.array([frame.wind_x, frame.wind_z], dtype=np.float32)
    return {
        "thrust": _scale_vector("thrust", thrust),
        "drag": _scale_vector("drag", drag),
        "gravity": _scale_vector("gravity", gravity),
        "net": _scale_vector("net", net),
        "wind": _scale_vector("wind", wind),
    }


def thrust_drag_magnitudes(config: FlightConfig, frame: DroneFrame) -> tuple[float, float]:
    thrust_total = float(sum(frame.motor_thrusts))
    rel_vx = frame.vx - frame.wind_x
    rel_vz = frame.vz - frame.wind_z
    drag_mag = float(np.hypot(config.drone.drag * rel_vx, config.drone.drag * rel_vz))
    return thrust_total, drag_mag


def _scale_vector(name: str, vector: np.ndarray) -> np.ndarray:
    style = FORCE_STYLES[name]
    scaled = vector * style["scale"]
    norm = float(np.hypot(scaled[0], scaled[1]))
    if norm > style["max_norm"]:
        scaled *= style["max_norm"] / norm
    return scaled
