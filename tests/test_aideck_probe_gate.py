from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from flightrl.semantic.aideck_probe_gate import (
    LabeledGray4Capture,
    evaluate_cross_capture_gray4,
)


ROOT = Path(__file__).resolve().parents[1]


def _frames(label: str, *, background: int = 17) -> np.ndarray:
    frames = np.full((8, 48, 64), background, dtype=np.uint8)
    if label == "positive":
        frames[:, 8:44, 25:39] = background + 136
        frames[1::2, 20:24, 5:12] = background + 17
    else:
        frames[:, 5:17, 5:59] = background + 136
        frames[1::2, 30:36, 45:52] = background + 17
    return frames


def _capture(tmp_path: Path, name: str, label: str, *, background: int = 17):
    path = tmp_path / f"{name}.npz"
    frames = _frames(label, background=background)
    metadata = {
        "schema": "flightrl.aideck_decoded_frame_capture.v2",
        "integrity_status": "unreviewed",
        "training_authority": False,
    }
    np.savez_compressed(
        path,
        decoded_frames=frames,
        host_time_s=np.arange(len(frames), dtype=np.float64) + 100.0,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return LabeledGray4Capture(
        label=label,
        frames=frames,
        indices=tuple(range(len(frames))),
        source=path,
        metadata=metadata,
    )


def test_cross_capture_gate_fails_closed_without_both_probe_classes(tmp_path) -> None:
    positive = _capture(tmp_path, "calibration-positive", "positive")
    negative = _capture(tmp_path, "calibration-negative", "negative")
    positive_probe = _capture(
        tmp_path, "positive-probe", "positive", background=51
    )

    report = evaluate_cross_capture_gray4(positive, negative, [positive_probe])

    assert report["positive_probe_passed"] is True
    assert report["negative_probe_passed"] is False
    assert report["cross_capture_gate_evaluable"] is False
    assert report["cross_capture_observability_passed"] is False
    assert report["semantic_generalization_authority"] is False


def test_cross_capture_gate_accepts_independent_probes_with_global_exposure_shift(
    tmp_path,
) -> None:
    positive = _capture(tmp_path, "calibration-positive", "positive")
    negative = _capture(tmp_path, "calibration-negative", "negative")
    probes = [
        _capture(tmp_path, "positive-probe", "positive", background=51),
        _capture(tmp_path, "negative-probe", "negative", background=51),
    ]

    report = evaluate_cross_capture_gray4(positive, negative, probes)

    assert report["cross_capture_gate_evaluable"] is True
    assert report["cross_capture_observability_passed"] is True
    assert all(probe["expected_recall"] == 1.0 for probe in report["probes"])
    assert report["training_authority"] is False


def test_cross_capture_cli_reports_missing_independent_negative(tmp_path) -> None:
    positive = _capture(tmp_path, "calibration-positive", "positive")
    negative = _capture(tmp_path, "calibration-negative", "negative")
    positive_probe = _capture(tmp_path, "positive-probe", "positive")
    output = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_aideck_cross_capture.py",
            "--positive-calibration",
            str(positive.source),
            "--negative-calibration",
            str(negative.source),
            "--positive-probe",
            str(positive_probe.source),
            "--sample-count",
            "8",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    report = json.loads(output.read_text())
    assert report["positive_probe_passed"] is True
    assert report["cross_capture_gate_evaluable"] is False
