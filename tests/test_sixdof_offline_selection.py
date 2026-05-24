from __future__ import annotations

from flightrl.sixdof.offline import OfflineTrainConfig, checkpoint_score, history_entry


def test_eval_selected_checkpoint_score_uses_yaw_when_safety_matches() -> None:
    config = OfflineTrainConfig(dataset="dummy", select_by_eval=True)
    aligned = checkpoint_payload(position_error=0.9, yaw_error=0.1, yaw_p95=0.3)
    drifting = checkpoint_payload(position_error=0.9, yaw_error=0.8, yaw_p95=1.6)

    assert checkpoint_score(aligned, config) < checkpoint_score(drifting, config)


def test_eval_selected_checkpoint_score_uses_saturation_as_tiebreaker() -> None:
    config = OfflineTrainConfig(dataset="dummy", select_by_eval=True)
    smooth = checkpoint_payload(action_saturation=0.0)
    saturated = checkpoint_payload(action_saturation=0.2)

    assert checkpoint_score(smooth, config) < checkpoint_score(saturated, config)


def test_history_entry_records_yaw_eval_metrics() -> None:
    entry = history_entry(
        3,
        0.2,
        0.3,
        {
            "mean_position_error_m": 0.4,
            "mean_yaw_error_rad": 0.1,
            "yaw_error_p95_rad": 0.3,
            "mean_completed_fraction": 1.0,
            "clearance_p01_m": 0.7,
            "min_clearance_m": 0.2,
        },
    )

    assert entry["eval_yaw_error_rad"] == 0.1
    assert entry["eval_yaw_error_p95_rad"] == 0.3


def checkpoint_payload(
    *,
    completed: float = 1.0,
    survival: float = 1.0,
    clearance: float = 0.5,
    position_error: float = 0.4,
    yaw_error: float = 0.2,
    yaw_p95: float = 0.5,
    action_saturation: float = 0.0,
) -> dict:
    return {
        "val_loss": 0.1,
        "selection_metrics": {
            "mean_completed_fraction": completed,
            "mean_survival_fraction": survival,
            "clearance_p01_m": clearance,
            "min_clearance_m": clearance,
            "mean_position_error_m": position_error,
            "mean_yaw_error_rad": yaw_error,
            "yaw_error_p95_rad": yaw_p95,
            "action_saturation_fraction": action_saturation,
        },
    }
