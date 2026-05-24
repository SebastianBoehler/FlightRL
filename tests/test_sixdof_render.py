from __future__ import annotations

import numpy as np
import subprocess
import sys

from flightrl.sixdof import AxisAlignedObstacle, BoxRoom
from flightrl.sixdof.render import render_rollout_html


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_render_rollout_html_writes_world_and_trajectory(tmp_path) -> None:
    room = BoxRoom(obstacles=(AxisAlignedObstacle(x_min=0.2, x_max=0.4, y_min=-0.1, y_max=0.1, z_min=0.0, z_max=0.6),))
    path = tmp_path / "rollout.html"
    trajectory = np.asarray([[0.0, 0.0, 0.2], [0.5, 0.1, 0.4]], dtype=np.float32)

    render_rollout_html(path, room=room, trajectory=trajectory)

    text = path.read_text()
    assert "FlightRL 6-DoF Rollout" in text
    assert "trajectory" in text
    assert "obstacle_0" in text


def test_render_sixdof_rollout_cli_writes_html(tmp_path) -> None:
    csv_path = tmp_path / "rollout.csv"
    csv_path.write_text("x,y,z\n0,0,0.2\n0.2,0.1,0.3\n")
    output = tmp_path / "rollout.html"

    subprocess.run(
        [sys.executable, "scripts/render_sixdof_rollout.py", "--input", str(csv_path), "--output", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.exists()
    assert "FlightRL 6-DoF Rollout" in output.read_text()
