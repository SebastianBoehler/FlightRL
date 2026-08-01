from __future__ import annotations

import numpy as np
import pytest
import torch

from flightrl.mujoco import is_mujoco_available
from flightrl.mujoco.door_observability import collect_synthetic_door_dataset
from flightrl.semantic.door_calibration import temporal_calibration_split
from flightrl.semantic.door_observability import (
    DoorObservabilityGate,
    DoorObservabilityNet,
    door_observability_model_from_state,
    labels_from_segmentation,
    observability_metrics,
)
from flightrl.semantic.door_observability_training import (
    DoorObservabilityTrainingConfig,
    train_door_observability,
)


def test_segmentation_label_uses_visible_door_pixels() -> None:
    segmentation = np.zeros((8, 10, 2), dtype=np.int32)
    segmentation[2:6, 3:8, 0] = 7

    label = labels_from_segmentation(segmentation, target_geom_id=7)

    assert label.visible == 1.0
    assert label.center_x == pytest.approx(0.55)
    assert label.center_y == pytest.approx(0.50)
    assert label.scale == pytest.approx(np.sqrt(20.0 / 80.0))


def test_segmentation_label_is_zero_when_door_is_absent() -> None:
    segmentation = np.zeros((8, 10, 2), dtype=np.int32)

    label = labels_from_segmentation(segmentation, target_geom_id=7)

    assert label.visible == 0.0
    assert label.center_x == 0.0
    assert label.center_y == 0.0
    assert label.scale == 0.0


def test_segmentation_label_combines_all_door_geometry() -> None:
    segmentation = np.zeros((8, 10, 2), dtype=np.int32)
    segmentation[2:6, 3:7, 0] = 7
    segmentation[3:5, 7:9, 0] = 8

    label = labels_from_segmentation(
        segmentation,
        target_geom_id=(7, 8),
    )

    assert label.visible == 1.0
    assert label.center_x == pytest.approx(0.60)
    assert label.scale == pytest.approx(0.5)


def test_tiny_observability_model_fits_deployment_budget() -> None:
    model = DoorObservabilityNet()
    output = model(torch.zeros((3, 1, 48, 64)))

    assert output.shape == (3, 4)
    assert sum(parameter.numel() for parameter in model.parameters()) < 50_000


def test_spatial_observability_model_preserves_checkpoint_shape() -> None:
    model = DoorObservabilityNet(pool_shape=(6, 8))
    restored = door_observability_model_from_state(model.state_dict())

    assert restored(torch.zeros((2, 1, 48, 64))).shape == (2, 4)
    assert restored.head[0].in_features == 32 * 6 * 8
    assert sum(parameter.numel() for parameter in restored.parameters()) < 120_000


def test_metrics_and_gate_pass_perfect_synthetic_predictions() -> None:
    labels = np.asarray(
        [
            [1.0, 0.25, 0.45, 0.30],
            [0.0, 0.00, 0.00, 0.00],
            [1.0, 0.75, 0.55, 0.20],
            [0.0, 0.00, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    predictions = labels.copy()
    predictions[:, 0] = np.asarray((0.99, 0.01, 0.98, 0.02))

    metrics = observability_metrics(predictions, labels)
    result = DoorObservabilityGate().evaluate(
        synthetic=metrics,
        real_positive=None,
        real_negative=None,
    )

    assert metrics.visibility_auroc == pytest.approx(1.0)
    assert metrics.centroid_median_error_widths == pytest.approx(0.0)
    assert result.synthetic_pass
    assert result.status == "incomplete_real_evidence"


def test_gate_fails_weak_synthetic_observability() -> None:
    weak = observability_metrics(
        np.asarray(
            [
                [0.40, 0.9, 0.9, 0.1],
                [0.60, 0.0, 0.0, 0.0],
                [0.45, 0.9, 0.9, 0.1],
                [0.55, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [
                [1.0, 0.2, 0.2, 0.2],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.8, 0.8, 0.2],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    result = DoorObservabilityGate().evaluate(
        synthetic=weak,
        real_positive=None,
        real_negative=None,
    )

    assert not result.synthetic_pass
    assert result.status == "failed_synthetic"


def test_mujoco_collection_produces_frame_contract_and_both_classes() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")

    dataset = collect_synthetic_door_dataset(
        room_seeds=(17,),
        samples_per_room=16,
        seed=29,
    )

    assert dataset.frames.shape == (16, 1, 48, 64)
    assert dataset.frames.dtype == np.float32
    assert dataset.labels.shape == (16, 4)
    assert np.all(np.isin(dataset.frames * 15.0, np.arange(16)))
    assert 0 < int(np.sum(dataset.labels[:, 0])) < 16


def test_observability_trainer_returns_finite_validation_predictions() -> None:
    frames = np.zeros((16, 1, 48, 64), dtype=np.float32)
    labels = np.zeros((16, 4), dtype=np.float32)
    frames[::2, :, 12:40, 24:40] = 1.0
    labels[::2] = np.asarray((1.0, 0.5, 0.54, 0.38), dtype=np.float32)

    result = train_door_observability(
        train_frames=frames,
        train_labels=labels,
        validation_frames=frames,
        validation_labels=labels,
        config=DoorObservabilityTrainingConfig(epochs=2, batch_size=8, seed=7),
        device="cpu",
    )

    assert result.validation_predictions.shape == labels.shape
    assert np.isfinite(result.validation_predictions).all()
    assert np.isfinite(result.final_train_loss)


def test_temporal_calibration_uses_only_first_class_segments() -> None:
    labels = np.asarray((1, 1, 1, 0, 0, 0), dtype=np.float32)
    scores = np.asarray((0.30, 0.20, 0.10, 0.01, 0.02, 0.03), dtype=np.float32)

    split = temporal_calibration_split(scores, labels, fraction=1.0 / 3.0)

    assert split.threshold == pytest.approx(0.155)
    assert np.flatnonzero(split.calibration_mask).tolist() == [0, 3]
    assert np.flatnonzero(split.evaluation_mask).tolist() == [1, 2, 4, 5]
