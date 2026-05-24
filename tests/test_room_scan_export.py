from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_room_scan_export_writes_ply_and_html(tmp_path: Path) -> None:
    log = tmp_path / "room.csv"
    log.write_text(
        "host_time_s,stateEstimate.x,stateEstimate.y,stateEstimate.z,stabilizer.roll,stabilizer.pitch,stabilizer.yaw,"
        "range.front,range.back,range.left,range.right,range.up,range.zrange\n"
        "10,1,2,0.40,0,0,0,1000,32766,500,32766,1200,400\n"
        "11,1.2,2.1,0.45,0,0,15,900,32766,600,32766,1100,450\n"
    )
    room = tmp_path / "room.json"
    room.write_text('{"room_estimate":{"x_min":0,"x_max":3,"y_min":0,"y_max":4,"z_min":0,"z_max":2.5}}\n')
    prefix = tmp_path / "scan"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_crazyflie_room_scan.py",
            "--input",
            str(log),
            "--room-report",
            str(room),
            "--output-prefix",
            str(prefix),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert prefix.with_suffix(".room.ply").exists()
    assert prefix.with_suffix(".room.html").exists()
    assert prefix.with_suffix(".room_points.csv").exists()
    assert "wrote" in result.stdout
