from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_sixdof_native_parity", ROOT / "scripts" / "build_sixdof_native_parity.py")
assert SPEC and SPEC.loader
PARITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PARITY
SPEC.loader.exec_module(PARITY)


def test_native_parity_cli_writes_readiness_compatible_report(tmp_path: Path) -> None:
    output = tmp_path / "native_parity.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_sixdof_native_parity.py"),
            "--reset-profile",
            "position_yaw_easy",
            "--num-envs",
            "8",
            "--steps",
            "4",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text())
    assert report["aligned"]["samples"] == 32
    assert report["aligned"]["signals"]["stateEstimate.x"]["rmse"] < 1e-5
    assert report["aligned"]["signals"]["range.front"]["rmse"] < 1.0
    assert output.with_suffix(".md").exists()


def test_native_parity_aggregate_uses_worst_profile() -> None:
    profiles = [
        profile("easy", state_rmse=0.1, range_rmse=0.2),
        profile("broad", state_rmse=0.3, range_rmse=0.1),
    ]
    aligned = PARITY.aggregate_aligned(profiles)

    assert aligned["signals"]["stateEstimate.x"]["rmse"] == 0.3
    assert aligned["signals"]["stateEstimate.x"]["worst_profile"] == "broad"
    assert aligned["signals"]["range.front"]["rmse"] == 0.2


def profile(name: str, *, state_rmse: float, range_rmse: float) -> dict:
    signals = {}
    for key in (*PARITY.STATE_SIGNALS, *PARITY.RANGE_SIGNALS):
        signals[key] = {"samples": 4, "rmse": range_rmse if key.startswith("range.") else state_rmse}
    return {"reset_profile": name, "samples": 4, "duration_s": 0.04, "signals": signals}
