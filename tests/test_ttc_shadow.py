from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from flightrl.hardware.ttc_shadow import evaluate_ttc_shadow_log


ROOT = Path(__file__).resolve().parents[1]


def test_ttc_shadow_report_groups_close_and_pinch_rows(tmp_path: Path) -> None:
    log = tmp_path / "ttc_shadow.csv"
    log.write_text(
        "range.front,range.back,range.left,range.right,range.zrange,min_horizontal_range_m,min_horizontal_ttc_s,"
        "raw_vx_m_s,raw_vy_m_s,raw_yawrate_deg_s,raw_zdistance_m,"
        "ttc_shadow_vx_m_s,ttc_shadow_vy_m_s,ttc_shadow_yawrate_deg_s,ttc_shadow_zdistance_m\n"
        "220,210,1800,600,500,0.21,0.50,0.0,0.6,0.0,0.5,0.0,0.55,0.0,0.5\n"
        "150,2000,2000,2000,500,0.15,0.20,-0.6,0.0,0.0,0.5,-0.55,0.0,0.0,0.5\n"
        "2000,2000,2000,2000,500,2.00,99.0,0.0,0.0,0.0,0.5,0.02,0.0,0.0,0.5\n"
    )

    report = evaluate_ttc_shadow_log(log)

    assert report["groups"]["all"]["samples"] == 3
    assert report["groups"]["pinch_like"]["samples"] == 1
    assert report["groups"]["close_lt_18cm"]["samples"] == 1
    assert report["groups"]["urgent_ttc_lt_35"]["samples"] == 1


def test_ttc_shadow_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    log = tmp_path / "ttc_shadow.csv"
    log.write_text(
        "range.front,range.back,range.left,range.right,range.zrange,min_horizontal_range_m,min_horizontal_ttc_s,"
        "vx_m_s,vy_m_s,yawrate_deg_s,zdistance_m,"
        "ttc_shadow_vx_m_s,ttc_shadow_vy_m_s,ttc_shadow_yawrate_deg_s,ttc_shadow_zdistance_m\n"
        "2000,2000,2000,2000,500,2.00,99.0,0.1,0.0,0.0,0.5,0.1,0.0,0.0,0.5\n"
    )
    output = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_ttc_shadow_log.py",
            "--input",
            str(log),
            "--output",
            str(output),
            "--target",
            "held",
        ],
        cwd=ROOT,
        check=True,
    )

    data = json.loads(output.read_text())
    assert data["groups"]["all"]["samples"] == 1
    assert output.with_suffix(".md").exists()
