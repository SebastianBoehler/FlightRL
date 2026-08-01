from __future__ import annotations

import csv
import subprocess
import sys

import pytest

from flightrl.sim2real.noise import summarize_stationary_noise, summarize_stationary_noise_logs


def test_stationary_noise_summary_accepts_stable_log(tmp_path) -> None:
    path = tmp_path / "stable.csv"
    write_noise_log(path, z_step=0.0001, roll_step=0.001)

    report = summarize_stationary_noise(path, min_duration_s=1.0)

    assert report["summary"]["stationary_noise_ready"] is True
    assert report["signals"]["stateEstimate.z"]["std"] > 0.0


def test_stationary_noise_summary_rejects_motion(tmp_path) -> None:
    path = tmp_path / "moving.csv"
    write_noise_log(path, z_step=0.02, roll_step=0.001)

    report = summarize_stationary_noise(path, min_duration_s=1.0, max_position_span_m=0.08)

    assert report["summary"]["stationary_noise_ready"] is False
    assert "position_motion" in report["summary"]["failures"]


def test_stationary_noise_summary_accepts_multiple_static_placements(tmp_path) -> None:
    first = tmp_path / "stable_a.csv"
    second = tmp_path / "stable_b.csv"
    write_noise_log(first, z_step=0.0001, roll_step=0.001, x_offset=0.0)
    write_noise_log(second, z_step=0.0001, roll_step=0.001, x_offset=1.0)

    report = summarize_stationary_noise_logs([first, second], min_duration_s=5.0, max_position_span_m=0.08)

    assert report["summary"]["stationary_noise_ready"] is True
    assert report["summary"]["inputs"] == 2
    assert report["summary"]["duration_s"] > 5.0
    assert report["summary"]["max_position_span_m"] < 0.08
    assert len(report["segments"]) == 2


def test_stationary_noise_uses_within_segment_range_noise(tmp_path) -> None:
    first = tmp_path / "range_a.csv"
    second = tmp_path / "range_b.csv"
    write_noise_log(first, z_step=0.0001, roll_step=0.001, range_offset=0.0)
    write_noise_log(second, z_step=0.0001, roll_step=0.001, range_offset=600.0)

    report = summarize_stationary_noise_logs([first, second], min_duration_s=5.0)
    front = report["signals"]["range.front"]

    assert report["summary"]["stationary_noise_ready"] is True
    assert front["std"] < 2.0
    assert front["span"] == 2.0
    assert front["global_span"] == 602.0


def test_stationary_noise_excludes_moving_range_segments(tmp_path) -> None:
    stable = tmp_path / "range_stable.csv"
    moving = tmp_path / "range_moving.csv"
    write_noise_log(stable, z_step=0.0001, roll_step=0.001)
    write_noise_log(moving, z_step=0.0001, roll_step=0.001, range_front_step=10.0)

    report = summarize_stationary_noise_logs([stable, moving], min_duration_s=5.0, max_range_span_mm=300.0)
    front = report["signals"]["range.front"]

    assert report["summary"]["stationary_noise_ready"] is True
    assert front["excluded_segments"] == 1
    assert front["excluded_samples"] == 60
    assert front["std"] < 2.0
    assert front["span"] == 2.0


def test_stationary_noise_filters_ranger_no_return_values(tmp_path) -> None:
    path = tmp_path / "range_no_return.csv"
    write_noise_log(path, z_step=0.0001, roll_step=0.001, range_no_return=True)

    report = summarize_stationary_noise(path, min_duration_s=1.0)
    front = report["signals"]["range.front"]

    assert report["summary"]["stationary_noise_ready"] is True
    assert front["raw_samples"] == 60
    assert front["invalid_samples"] == 30
    assert front["samples"] == 30
    assert front["std"] < 2.0


@pytest.mark.parametrize(("field", "value"), (("host_time_s", "nan"), ("acc.x", "inf")))
def test_stationary_noise_rejects_nonfinite_samples(
    tmp_path,
    field: str,
    value: str,
) -> None:
    path = tmp_path / "corrupt.csv"
    write_noise_log(path, z_step=0.0001, roll_step=0.001)
    rows = list(csv.DictReader(path.open()))
    rows[20][field] = value
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = summarize_stationary_noise(path, min_duration_s=1.0)

    assert report["summary"]["stationary_noise_ready"] is False
    assert "nonfinite_values" in report["summary"]["failures"]


def test_stationary_noise_rejects_reordered_timestamps(tmp_path) -> None:
    path = tmp_path / "reordered.csv"
    write_noise_log(path, z_step=0.0001, roll_step=0.001)
    rows = list(csv.DictReader(path.open()))
    rows[20]["host_time_s"] = rows[19]["host_time_s"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = summarize_stationary_noise(path, min_duration_s=1.0)

    assert report["summary"]["stationary_noise_ready"] is False
    assert "time_monotonic" in report["summary"]["failures"]


def test_stationary_noise_rejects_sparse_low_rate_evidence(tmp_path) -> None:
    path = tmp_path / "sparse.csv"
    fields = ["host_time_s", *noise_columns()]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: 0.0 for field in fields})
        writer.writerow({**{field: "" for field in fields}, "host_time_s": 30.0})

    report = summarize_stationary_noise(path)

    assert report["summary"]["stationary_noise_ready"] is False
    assert {"rows", "sample_rate", "signal_samples", "signal_valid_ratio"}.issubset(
        report["summary"]["failures"]
    )


def test_stationary_noise_logs_validate_direct_thresholds_and_inputs() -> None:
    with pytest.raises(ValueError, match="at least one input"):
        summarize_stationary_noise_logs([])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        summarize_stationary_noise_logs([], min_duration_s=float("nan"))


def test_stationary_noise_cli_writes_report(tmp_path) -> None:
    path = tmp_path / "stable.csv"
    output = tmp_path / "summary.json"
    write_noise_log(path, z_step=0.0001, roll_step=0.001)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_stationary_noise.py",
            "--input",
            str(path),
            "--output",
            str(output),
            "--min-duration-s",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "stationary_noise_ready=True" in result.stdout
    assert output.exists()
    assert output.with_suffix(".md").exists()


def test_stationary_noise_cli_accepts_repeated_inputs(tmp_path) -> None:
    first = tmp_path / "stable_a.csv"
    second = tmp_path / "stable_b.csv"
    output = tmp_path / "summary.json"
    write_noise_log(first, z_step=0.0001, roll_step=0.001, x_offset=0.0)
    write_noise_log(second, z_step=0.0001, roll_step=0.001, x_offset=1.0)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_stationary_noise.py",
            "--input",
            str(first),
            "--input",
            str(second),
            "--output",
            str(output),
            "--min-duration-s",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "stationary_noise_ready=True" in result.stdout
    assert output.exists()


def write_noise_log(
    path,
    *,
    z_step: float,
    roll_step: float,
    x_offset: float = 0.0,
    range_no_return: bool = False,
    range_offset: float = 0.0,
    range_front_step: float = 0.0,
) -> None:
    fields = ["host_time_s", *noise_columns()]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx in range(60):
            writer.writerow(
                {
                    field: value_for(field, idx, z_step=z_step, roll_step=roll_step, x_offset=x_offset, range_no_return=range_no_return)
                    + (range_offset if field.startswith("range.") and not (field == "range.front" and range_no_return and idx % 2) else 0.0)
                    + (idx * range_front_step if field == "range.front" else 0.0)
                    for field in fields
                }
            )


def noise_columns() -> list[str]:
    return [
        "stateEstimate.x",
        "stateEstimate.y",
        "stateEstimate.z",
        "stabilizer.roll",
        "stabilizer.pitch",
        "stabilizer.yaw",
        "acc.x",
        "acc.y",
        "acc.z",
        "gyro.x",
        "gyro.y",
        "gyro.z",
        "range.front",
        "range.back",
        "range.left",
        "range.right",
        "range.up",
        "range.zrange",
    ]


def value_for(field: str, idx: int, *, z_step: float, roll_step: float, x_offset: float, range_no_return: bool) -> float:
    if field == "host_time_s":
        return idx * 0.05
    if field == "stateEstimate.x":
        return x_offset
    if field == "stateEstimate.z":
        return 0.4 + idx * z_step
    if field == "stabilizer.roll":
        return idx * roll_step
    if field == "range.front" and range_no_return and idx % 2:
        return 32766.0
    if field.startswith("range."):
        return 1000.0 + (idx % 3)
    return 0.0
