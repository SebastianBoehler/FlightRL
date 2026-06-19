from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_compare_deterministic_avoidance_baselines(tmp_path: Path) -> None:
    log = tmp_path / "log.csv"
    log.write_text(
        "\n".join(
            [
                "range.front,range.back,range.left,range.right,range.up,range.zrange,vx_m_s,vy_m_s,yawrate_deg_s,zdistance_m",
                "150,3000,3000,3000,2000,500,-0.25,0.0,0.0,0.5",
                "3000,3000,180,3000,2000,500,0.0,-0.25,0.0,0.5",
            ]
        )
        + "\n"
    )
    output = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_deterministic_avoidance_baselines.py"),
            "--input",
            f"{log}:0:0.2",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )

    report = json.loads(output.read_text())
    assert report["rows"] == 2
    assert report["best_by_close_escape"]["close_escape_agreement"] == 1.0
    assert output.with_suffix(".md").exists()
