from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import evaluate_aideck_grounding as evaluate


ROOT = Path(__file__).resolve().parents[1]


def _write_capture(path, frames: np.ndarray) -> None:
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


def test_npz_capture_is_sampled_without_manual_png_conversion(tmp_path) -> None:
    frames = np.stack(
        tuple(np.full((48, 64), 17 * level, dtype=np.uint8) for level in range(4))
    )
    metadata = {
        "schema": "flightrl.aideck_decoded_frame_capture.v2",
        "integrity_status": "unreviewed",
        "training_authority": False,
    }
    capture = tmp_path / "decoded_frames.npz"
    np.savez_compressed(
        capture,
        decoded_frames=frames,
        host_time_s=np.asarray((10.0, 11.0, 12.0, 13.0)),
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    archived = evaluate.load_archived_frames(capture, limit=3)

    assert [frame.index for frame in archived] == [0, 1, 3]
    assert [int(frame.pixels[0, 0]) for frame in archived] == [0, 17, 51]
    assert [frame.host_time_s for frame in archived] == [10.0, 11.0, 13.0]
    assert all(frame.source == capture for frame in archived)
    assert archived[0].capture_metadata == metadata


def test_paired_gate_separates_operator_labeled_gray4_scenes(tmp_path) -> None:
    positive = np.full((8, 48, 64), 17, dtype=np.uint8)
    negative = np.full((8, 48, 64), 17, dtype=np.uint8)
    positive[:, 8:44, 25:39] = 153
    negative[:, 5:17, 5:59] = 153
    positive[1::2, 20:24, 5:12] = 34
    negative[1::2, 30:36, 45:52] = 34
    positive_path = tmp_path / "positive.npz"
    negative_path = tmp_path / "negative.npz"
    _write_capture(positive_path, positive)
    _write_capture(negative_path, negative)

    report = evaluate.evaluate_paired_captures(
        positive_path,
        negative_path,
        sample_count=8,
    )

    assert report["schema"] == "flightrl.aideck_paired_observability.v1"
    assert report["paired_observability_passed"] is True
    assert report["balanced_accuracy"] == 1.0
    assert report["minimum_classification_margin"] >= 0.10
    assert report["nonspatial_histogram_baseline"]["balanced_accuracy"] == 1.0
    assert report["scene_confound_detected"] is True
    assert report["shortcut_resistant_semantic_calibration_passed"] is False
    assert report["input_contract"] == {
        "shape": [48, 64],
        "dtype": "uint8",
        "levels": "decoded_gray4_multiples_of_17",
        "model_normalization": "float32(nibble) / 15",
    }
    assert report["training_authority"] is False
    assert report["semantic_generalization_authority"] is False


def test_paired_gate_rejects_non_gray4_capture(tmp_path) -> None:
    positive = np.zeros((4, 48, 64), dtype=np.uint8)
    positive[0, 0, 0] = 1
    negative = np.zeros((4, 48, 64), dtype=np.uint8)
    positive_path = tmp_path / "positive.npz"
    negative_path = tmp_path / "negative.npz"
    _write_capture(positive_path, positive)
    _write_capture(negative_path, negative)

    with pytest.raises(ValueError, match="exact decoded gray4"):
        evaluate.evaluate_paired_captures(
            positive_path,
            negative_path,
            sample_count=4,
        )


def test_paired_gate_cli_writes_non_authoritative_report(tmp_path) -> None:
    positive = np.full((4, 48, 64), 17, dtype=np.uint8)
    negative = np.full((4, 48, 64), 17, dtype=np.uint8)
    positive[:, 8:44, 25:39] = 153
    negative[:, 5:17, 5:59] = 153
    positive_path = tmp_path / "positive.npz"
    negative_path = tmp_path / "negative.npz"
    output = tmp_path / "report.json"
    _write_capture(positive_path, positive)
    _write_capture(negative_path, negative)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_aideck_pair.py",
            "--positive",
            str(positive_path),
            "--negative",
            str(negative_path),
            "--sample-count",
            "4",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["paired_observability_passed"] is True
    assert report["deployment_authority"] is False
