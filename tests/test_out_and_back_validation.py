from __future__ import annotations

import csv
import json
from pathlib import Path


PHASES = {
    "takeoff": 1.0,
    "forward": 6.0,
    "backward": 12.0,
    "land": 18.0,
    "complete": 22.0,
}


def write_run(root: Path, *, backward_distance_m: float = 0.5) -> None:
    root.mkdir()
    with (root / "events.jsonl").open("w") as handle:
        for phase, host_time_s in PHASES.items():
            handle.write(json.dumps({"phase": phase, "host_time_s": host_time_s}) + "\n")
    with (root / "telemetry.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "host_time_s",
                "crazyflie_time_ms",
                "stateEstimate.x",
                "stateEstimate.y",
                "stateEstimate.z",
                "stateEstimate.yaw",
                "pm.vbat",
                "pm.state",
                "stateEstimate.roll",
                "stateEstimate.pitch",
            )
        )
        for index in range(441):
            t = index * 0.05
            z = min(0.4, max(0.0, (t - 1.0) * 0.1))
            x = 0.0
            if t >= 6.0:
                x = 0.5 * min(1.0, (t - 6.0) / 5.0)
            if t >= 12.0:
                x = 0.5 - backward_distance_m * min(1.0, (t - 12.0) / 5.0)
            if t >= 18.0:
                z = max(0.0, 0.4 - (t - 18.0) * 0.1)
            writer.writerow((t, index * 50, x, 0.0, z, 0.0, 3.6, 0, 1.0, -2.0))


def test_out_and_back_validation_accepts_symmetric_return(tmp_path: Path) -> None:
    from flightrl.hardware.flight_validation import validate_out_and_back

    run_dir = tmp_path / "passing"
    write_run(run_dir)

    report = validate_out_and_back(run_dir)

    assert report["out_and_back_passed"] is True
    assert report["metrics"]["forward"]["distance_m"] == 0.5
    assert report["metrics"]["backward"]["distance_m"] == 0.5
    assert report["metrics"]["return_error_m"] == 0.0
    assert report["metrics"]["maximum_abs_roll_deg"] == 1.0
    assert report["metrics"]["maximum_abs_pitch_deg"] == 2.0


def test_out_and_back_validation_rejects_asymmetric_return(tmp_path: Path) -> None:
    from flightrl.hardware.flight_validation import validate_out_and_back

    run_dir = tmp_path / "asymmetric"
    write_run(run_dir, backward_distance_m=0.1)

    report = validate_out_and_back(run_dir)

    assert report["out_and_back_passed"] is False
    assert report["checks"]["backward"] is False
    assert report["checks"]["repeatability"] is False
    assert report["checks"]["returned"] is False
