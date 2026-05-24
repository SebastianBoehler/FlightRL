from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from .geometry import AxisAlignedObstacle, BoxRoom


def render_rollout_html(path: str | Path, *, room: BoxRoom, trajectory: np.ndarray) -> Path:
    output = Path(path)
    fig = go.Figure()
    add_box_edges(fig, room, "room", "#2f3542")
    for index, obstacle in enumerate(room.obstacles):
        add_box_edges(fig, obstacle, f"obstacle_{index}", "#d9480f")
    if len(trajectory):
        fig.add_trace(
            go.Scatter3d(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=trajectory[:, 2],
                mode="lines+markers",
                name="trajectory",
                line={"color": "#1c7ed6", "width": 5},
                marker={"size": 3},
            )
        )
    fig.update_layout(
        title="FlightRL 6-DoF Rollout",
        scene={"xaxis_title": "x m", "yaxis_title": "y m", "zaxis_title": "z m", "aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs="cdn")
    return output


def add_box_edges(fig: go.Figure, box: AxisAlignedObstacle, name: str, color: str) -> None:
    corners = box_corners(box)
    edges = ((0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for start, end in edges:
        for idx in (start, end):
            x.append(float(corners[idx, 0]))
            y.append(float(corners[idx, 1]))
            z.append(float(corners[idx, 2]))
        x.append(None)
        y.append(None)
        z.append(None)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", name=name, line={"color": color, "width": 3}))


def box_corners(box: AxisAlignedObstacle) -> np.ndarray:
    return np.asarray(
        [
            [box.x_min, box.y_min, box.z_min],
            [box.x_max, box.y_min, box.z_min],
            [box.x_min, box.y_max, box.z_min],
            [box.x_max, box.y_max, box.z_min],
            [box.x_min, box.y_min, box.z_max],
            [box.x_max, box.y_min, box.z_max],
            [box.x_min, box.y_max, box.z_max],
            [box.x_max, box.y_max, box.z_max],
        ],
        dtype=np.float32,
    )
