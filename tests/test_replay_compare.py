from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_replay_compare_aligns_common_signals(tmp_path: Path) -> None:
    real = tmp_path / "real.csv"
    sim = tmp_path / "sim.csv"
    output = tmp_path / "compare.json"
    real.write_text(
        "host_time_s,stateEstimate.z,range.front,vx_m_s\n"
        "100,0.1,1000,0.0\n"
        "101,0.2,900,0.1\n"
        "102,0.3,800,0.2\n"
    )
    sim.write_text(
        "host_time_s,stateEstimate.z,range.front,vx_m_s\n"
        "0,0.1,1000,0.0\n"
        "2,0.5,600,0.4\n"
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_crazyflie_replay.py",
            "--real",
            str(real),
            "--sim",
            str(sim),
            "--align-time",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["aligned"]["samples"] == 3
    assert report["aligned"]["overlap_duration_s"] == 2.0
    assert set(report["aligned"]["signals"]) == {"stateEstimate.z", "range.front", "vx_m_s"}
    assert report["aligned"]["signals"]["stateEstimate.z"]["rmse"] > 0.0


def test_replay_compare_can_limit_aligned_signals(tmp_path: Path) -> None:
    real = tmp_path / "real.csv"
    sim = tmp_path / "sim.csv"
    output = tmp_path / "compare.json"
    text = "host_time_s,stateEstimate.z,range.front\n0,0.1,1000\n1,0.2,900\n"
    real.write_text(text)
    sim.write_text(text)
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_crazyflie_replay.py",
            "--real",
            str(real),
            "--sim",
            str(sim),
            "--align-time",
            "--signals",
            "range.front",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert list(report["aligned"]["signals"]) == ["range.front"]
    assert report["aligned"]["signals"]["range.front"]["rmse"] == 0.0


def test_replay_compare_filters_invalid_range_sentinels(tmp_path: Path) -> None:
    real = tmp_path / "real.csv"
    sim = tmp_path / "sim.csv"
    output = tmp_path / "compare.json"
    real.write_text("host_time_s,range.front\n0,1000\n1,32766\n2,800\n")
    sim.write_text("host_time_s,range.front\n0,1000\n1,2000\n2,800\n")
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_crazyflie_replay.py",
            "--real",
            str(real),
            "--sim",
            str(sim),
            "--align-time",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    signal = json.loads(output.read_text())["aligned"]["signals"]["range.front"]
    assert signal["samples"] == 2
    assert signal["rmse"] == 0.0
