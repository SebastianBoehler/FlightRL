from __future__ import annotations

import numpy as np

from flightrl.replay import fit_linear_calibration, fit_signal, signal_error


def test_fit_signal_recovers_scale_and_bias() -> None:
    sim = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    real = 2.0 * sim + 5.0
    fit = fit_signal(real, sim)
    assert fit["scale"] == 2.0
    assert fit["bias"] == 5.0
    assert fit["fitted_rmse"] < 1e-6


def test_fit_linear_calibration_uses_aligned_valid_range_samples() -> None:
    real_rows = [
        {"host_time_s": "10", "range.front": "1000"},
        {"host_time_s": "11", "range.front": "32766"},
        {"host_time_s": "12", "range.front": "3000"},
    ]
    sim_rows = [
        {"host_time_s": "0", "range.front": "500"},
        {"host_time_s": "1", "range.front": "1000"},
        {"host_time_s": "2", "range.front": "1500"},
    ]
    report = fit_linear_calibration(real_rows, sim_rows)
    fit = report["signals"]["range.front"]
    assert fit["samples"] == 2
    assert fit["scale"] == 2.0
    assert abs(fit["bias"]) < 1e-5


def test_signal_error_wraps_yaw_degrees() -> None:
    real = np.asarray([350.0, 10.0, -179.0], dtype=np.float32)
    sim = np.asarray([10.0, 350.0, 179.0], dtype=np.float32)

    error = signal_error("stabilizer.yaw", real, sim)

    assert np.allclose(error, [20.0, -20.0, -2.0])
