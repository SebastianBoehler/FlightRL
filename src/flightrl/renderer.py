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
    vx: float
    vz: float
    ax: float
    az: float
    pitch: float
    pitch_rate: float
    target_x: float
    target_z: float
    distance: float
    reward_total: float
    motor_left: float
    motor_right: float
    command_0: float
    command_1: float
    active_target: int
    target_count: int


class FlightRenderer:
    def __init__(self, config: FlightConfig, mode: str, fps: float | None = None):
        self.config = config
        self.mode = mode
        self.fps = fps or 30.0
        self._figure: Figure | None = None
        self._canvas: FigureCanvasAgg | None = None
        self._world_ax = None
        self._hud_ax = None
        self._artists: dict[str, object] = {}

    def render(self, frame: DroneFrame) -> np.ndarray | None:
        self._ensure_figure()
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
        self._world_ax = None
        self._hud_ax = None
        self._artists.clear()

    def _ensure_figure(self) -> None:
        if self._figure is not None:
            return
        if self.mode == "human":
            plt.ion()
            self._figure, axes = plt.subplots(
                1,
                2,
                figsize=(9.5, 5.8),
                gridspec_kw={"width_ratios": [3.4, 1.5]},
                tight_layout=True,
            )
            self._canvas = self._figure.canvas
        else:
            self._figure = Figure(figsize=(9.5, 5.8), tight_layout=True)
            self._canvas = FigureCanvasAgg(self._figure)
            axes = self._figure.subplots(1, 2, gridspec_kw={"width_ratios": [3.4, 1.5]})
        self._figure.patch.set_facecolor("#f7f8fa")
        self._world_ax, self._hud_ax = axes
        self._setup_world()
        self._setup_hud()

    def _setup_world(self) -> None:
        assert self._world_ax is not None
        ax = self._world_ax
        drone = self.config.drone
        ax.set_facecolor("#fcfcfb")
        ax.set_xlim(-drone.x_limit, drone.x_limit)
        ax.set_ylim(drone.floor_z, drone.z_limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Horizontal Position")
        ax.set_ylabel("Altitude")
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.22, color="#607080")
        ax.axhline(drone.floor_z, color="#9aa4ad", linestyle="-", linewidth=1.0, alpha=0.9)
        self._artists["target"] = ax.scatter([], [], s=160, c="#9c6644", marker="x", linewidths=2.0, zorder=6)
        self._artists["target_ring"] = ax.scatter([], [], s=820, facecolors="none", edgecolors="#d7b48a", linewidths=1.0, alpha=0.85, zorder=4)
        self._artists["trail"], = ax.plot([], [], color="#94a3b8", linewidth=1.6, alpha=0.5, zorder=2)
        self._artists["link"], = ax.plot([], [], color="#b8c0c8", linewidth=1.1, linestyle="--", alpha=0.9, zorder=3)
        self._artists["body"], = ax.plot([], [], color="#20262e", linewidth=3.2, solid_capstyle="round", zorder=8)
        self._artists["cross_arm"], = ax.plot([], [], color="#6b7280", linewidth=2.0, alpha=0.9, zorder=7)
        self._artists["forward"], = ax.plot([], [], color="#4f6d7a", linewidth=1.8, zorder=7)
        self._artists["thrust"], = ax.plot([], [], color="#2f7f7a", linewidth=2.2, zorder=7)
        self._artists["velocity"], = ax.plot([], [], color="#5b6c7d", linewidth=1.6, zorder=7)
        self._artists["acceleration"], = ax.plot([], [], color="#b26a4f", linewidth=1.5, linestyle=":", zorder=7)
        self._artists["left_front_plume"], = ax.plot([], [], color="#2f7f7a", linewidth=1.2, alpha=0.7, zorder=5)
        self._artists["left_rear_plume"], = ax.plot([], [], color="#2f7f7a", linewidth=1.2, alpha=0.7, zorder=5)
        self._artists["right_front_plume"], = ax.plot([], [], color="#2f7f7a", linewidth=1.2, alpha=0.7, zorder=5)
        self._artists["right_rear_plume"], = ax.plot([], [], color="#2f7f7a", linewidth=1.2, alpha=0.7, zorder=5)
        self._artists["rotors"] = ax.scatter(
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            s=[110, 110, 110, 110],
            c=["#f8fafc", "#f8fafc", "#f8fafc", "#f8fafc"],
            edgecolors="#48525c",
            linewidths=1.0,
            zorder=9,
        )
        self._artists["history_x"] = []
        self._artists["history_z"] = []

    def _setup_hud(self) -> None:
        assert self._hud_ax is not None
        ax = self._hud_ax
        ax.set_facecolor("#f1f4f7")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#d9e0e6")
            spine.set_linewidth(0.8)
        ax.plot([0.08, 0.92], [0.82, 0.82], color="#d9e0e6", linewidth=0.8)
        ax.plot([0.08, 0.92], [0.42, 0.42], color="#d9e0e6", linewidth=0.8)
        self._artists["hud_title"] = ax.text(0.08, 0.93, "", color="#1f2933", fontsize=13, fontweight="bold")
        self._artists["hud_task"] = ax.text(0.08, 0.875, "", color="#52606d", fontsize=9.5)
        self._artists["hud_metrics"] = ax.text(0.08, 0.77, "", color="#24323f", fontsize=9.6, linespacing=1.42, va="top")
        self._artists["hud_motor_label"] = ax.text(0.08, 0.37, "Rotor Pair Thrust", color="#1f2933", fontsize=10, fontweight="bold")
        self._artists["hud_command_label"] = ax.text(0.08, 0.19, "Control Command", color="#1f2933", fontsize=10, fontweight="bold")
        motor_bars = ax.barh([0.31, 0.25], [0.0, 0.0], left=0.18, height=0.028, color=["#607d8b", "#90a4ae"], alpha=0.95)
        command_bars = ax.barh([0.12, 0.07], [0.0, 0.0], left=0.54, height=0.028, color=["#6b9080", "#a68a64"], alpha=0.95)
        ax.plot([0.18, 0.86], [0.31, 0.31], color="#d3dbe3", linewidth=6.0, solid_capstyle="butt", zorder=0)
        ax.plot([0.18, 0.86], [0.25, 0.25], color="#d3dbe3", linewidth=6.0, solid_capstyle="butt", zorder=0)
        ax.plot([0.20, 0.88], [0.12, 0.12], color="#d3dbe3", linewidth=6.0, solid_capstyle="butt", zorder=0)
        ax.plot([0.20, 0.88], [0.07, 0.07], color="#d3dbe3", linewidth=6.0, solid_capstyle="butt", zorder=0)
        ax.plot([0.54, 0.54], [0.03, 0.16], color="#9aa5b1", linewidth=0.9)
        ax.text(0.08, 0.31, "Left Pair", color="#52606d", fontsize=8.8, va="center")
        ax.text(0.08, 0.25, "Right Pair", color="#52606d", fontsize=8.8, va="center")
        ax.text(0.08, 0.12, "Collective", color="#52606d", fontsize=8.8, va="center")
        ax.text(0.08, 0.07, "Pitch", color="#52606d", fontsize=8.8, va="center")
        self._artists["motor_bars"] = motor_bars.patches
        self._artists["command_bars"] = command_bars.patches

    def _draw_frame(self, frame: DroneFrame) -> None:
        assert self._world_ax is not None
        arm = self.config.drone.arm_length
        max_thrust = 0.5 * self.config.drone.max_total_thrust
        history_x = self._artists["history_x"]
        history_z = self._artists["history_z"]
        history_x.append(frame.x)
        history_z.append(frame.z)
        if len(history_x) > 180:
            del history_x[0]
            del history_z[0]

        fx = np.cos(frame.pitch)
        fz = np.sin(frame.pitch)
        tx = -np.sin(frame.pitch)
        tz = np.cos(frame.pitch)
        dx = arm * fx
        dz = arm * fz
        left_pair = (frame.x - dx, frame.z - dz)
        right_pair = (frame.x + dx, frame.z + dz)
        depth = 0.23 * arm
        front_offset = (depth * tx, depth * tz)
        rear_offset = (-depth * tx, -depth * tz)
        left_front = (left_pair[0] + front_offset[0], left_pair[1] + front_offset[1])
        left_rear = (left_pair[0] + rear_offset[0], left_pair[1] + rear_offset[1])
        right_front = (right_pair[0] + front_offset[0], right_pair[1] + front_offset[1])
        right_rear = (right_pair[0] + rear_offset[0], right_pair[1] + rear_offset[1])

        rotor_scale = np.array([frame.motor_left, frame.motor_right], dtype=np.float32) / max(max_thrust, 1e-6)
        rotor_scale = np.clip(rotor_scale, 0.0, 1.0)
        rotor_sizes = [
            85.0 + 45.0 * float(rotor_scale[0]),
            85.0 + 45.0 * float(rotor_scale[0]),
            85.0 + 45.0 * float(rotor_scale[1]),
            85.0 + 45.0 * float(rotor_scale[1]),
        ]

        self._artists["target"].set_offsets([[frame.target_x, frame.target_z]])
        self._artists["target_ring"].set_offsets([[frame.target_x, frame.target_z]])
        self._artists["trail"].set_data(history_x, history_z)
        self._artists["link"].set_data([frame.x, frame.target_x], [frame.z, frame.target_z])
        self._artists["body"].set_data([left_pair[0], right_pair[0]], [left_pair[1], right_pair[1]])
        self._artists["cross_arm"].set_data(
            [frame.x - 0.55 * arm * tx, frame.x + 0.55 * arm * tx],
            [frame.z - 0.55 * arm * tz, frame.z + 0.55 * arm * tz],
        )
        self._artists["forward"].set_data([frame.x, frame.x + 1.65 * dx], [frame.z, frame.z + 1.65 * dz])
        self._artists["thrust"].set_data([frame.x, frame.x + 1.15 * tx], [frame.z, frame.z + 1.15 * tz])
        self._artists["velocity"].set_data([frame.x, frame.x + 0.75 * frame.vx], [frame.z, frame.z + 0.75 * frame.vz])
        self._artists["acceleration"].set_data([frame.x, frame.x + 0.2 * frame.ax], [frame.z, frame.z + 0.2 * frame.az])
        left_plume = 0.28 + 0.42 * float(rotor_scale[0])
        right_plume = 0.28 + 0.42 * float(rotor_scale[1])
        self._artists["left_front_plume"].set_data(
            [left_front[0], left_front[0] + left_plume * tx],
            [left_front[1], left_front[1] + left_plume * tz],
        )
        self._artists["left_rear_plume"].set_data(
            [left_rear[0], left_rear[0] + left_plume * tx],
            [left_rear[1], left_rear[1] + left_plume * tz],
        )
        self._artists["right_front_plume"].set_data(
            [right_front[0], right_front[0] + right_plume * tx],
            [right_front[1], right_front[1] + right_plume * tz],
        )
        self._artists["right_rear_plume"].set_data(
            [right_rear[0], right_rear[0] + right_plume * tx],
            [right_rear[1], right_rear[1] + right_plume * tz],
        )
        self._artists["rotors"].set_offsets([left_front, left_rear, right_front, right_rear])
        self._artists["rotors"].set_sizes(rotor_sizes)
        self._artists["rotors"].set_facecolor(["#f8fafc", "#f8fafc", "#f8fafc", "#f8fafc"])
        self._world_ax.set_title(
            f"{self.config.task.task_type} | target {frame.active_target}/{frame.target_count} | distance {frame.distance:.2f}",
            fontsize=12.5,
            pad=12,
            color="#253039",
        )
        self._draw_hud(frame, rotor_scale)

    def _draw_hud(self, frame: DroneFrame, rotor_scale: np.ndarray) -> None:
        speed = float(np.hypot(frame.vx, frame.vz))
        accel = float(np.hypot(frame.ax, frame.az))
        self._artists["hud_title"].set_text("Flight Inspection")
        self._artists["hud_task"].set_text(
            f"{self.config.environment.action_mode} | dt={self.config.environment.dt:.3f}s"
        )
        self._artists["hud_metrics"].set_text(
            "\n".join(
                [
                    f"pitch      {np.degrees(frame.pitch):6.1f} deg",
                    f"pitch rate {np.degrees(frame.pitch_rate):6.1f} deg/s",
                    f"speed      {speed:6.2f} m/s",
                    f"accel      {accel:6.2f} m/s²",
                    f"reward     {frame.reward_total:6.2f}",
                    f"distance   {frame.distance:6.2f} m",
                ]
            )
        )
        for patch, value in zip(self._artists["motor_bars"], rotor_scale, strict=True):
            patch.set_x(0.18)
            patch.set_width(0.68 * float(value))
        for patch, value in zip(self._artists["command_bars"], [frame.command_0, frame.command_1], strict=True):
            clipped = float(np.clip(value, -1.0, 1.0))
            patch.set_x(0.54 if clipped >= 0.0 else 0.54 + 0.34 * clipped)
            patch.set_width(0.34 * abs(clipped))
