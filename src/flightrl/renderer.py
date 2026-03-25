from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .config import FlightConfig


@dataclass(slots=True)
class DroneFrame:
    x: float
    z: float
    pitch: float
    target_x: float
    target_z: float
    distance: float
    reward_total: float


class FlightRenderer:
    def __init__(self, config: FlightConfig, mode: str, fps: float | None = None):
        self.config = config
        self.mode = mode
        self.fps = fps or 30.0
        self._figure: Figure | None = None
        self._canvas: FigureCanvasAgg | None = None
        self._axes = None
        self._artists: dict[str, object] = {}

    def render(self, frame: DroneFrame) -> np.ndarray | None:
        self._ensure_figure()
        assert self._axes is not None
        self._draw_frame(frame)
        self._canvas.draw()
        if self.mode == "human":
            plt.show(block=False)
            plt.pause(max(1.0 / self.fps, 0.001))
            return None
        buffer = np.asarray(self._canvas.buffer_rgba(), dtype=np.uint8)
        return np.ascontiguousarray(buffer[..., :3])

    def close(self) -> None:
        if self._figure is not None:
            plt.close(self._figure)
        self._figure = None
        self._canvas = None
        self._axes = None
        self._artists.clear()

    def _ensure_figure(self) -> None:
        if self._figure is not None:
            return
        if self.mode == "human":
            plt.ion()
            self._figure, self._axes = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
            self._canvas = self._figure.canvas
        else:
            self._figure = Figure(figsize=(6, 6), tight_layout=True)
            self._canvas = FigureCanvasAgg(self._figure)
            self._axes = self._figure.add_subplot(1, 1, 1)
        ax = self._axes
        drone = self.config.drone
        ax.set_xlim(-drone.x_limit, drone.x_limit)
        ax.set_ylim(drone.floor_z, drone.z_limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.5)
        ax.axhline(drone.floor_z, color="tab:red", linestyle="--", linewidth=1.2)
        self._artists["target"] = ax.scatter([], [], s=70, c="tab:orange", marker="x", linewidths=2)
        self._artists["trail"], = ax.plot([], [], color="tab:blue", linewidth=1.5, alpha=0.6)
        self._artists["body"], = ax.plot([], [], color="tab:blue", linewidth=3.0)
        self._artists["heading"], = ax.plot([], [], color="tab:green", linewidth=2.0)
        self._artists["history_x"] = []
        self._artists["history_z"] = []

    def _draw_frame(self, frame: DroneFrame) -> None:
        ax = self._axes
        artists = self._artists
        assert ax is not None
        arm = self.config.drone.arm_length
        dx = arm * np.cos(frame.pitch)
        dz = arm * np.sin(frame.pitch)
        artists["target"].set_offsets([[frame.target_x, frame.target_z]])
        history_x = artists["history_x"]
        history_z = artists["history_z"]
        history_x.append(frame.x)
        history_z.append(frame.z)
        artists["trail"].set_data(history_x, history_z)
        artists["body"].set_data([frame.x - dx, frame.x + dx], [frame.z - dz, frame.z + dz])
        artists["heading"].set_data([frame.x, frame.x + 0.8 * dx], [frame.z, frame.z + 0.8 * dz])
        ax.set_title(
            f"{self.config.task.task_type} | "
            f"distance={frame.distance:.2f} | reward={frame.reward_total:.2f}"
        )
