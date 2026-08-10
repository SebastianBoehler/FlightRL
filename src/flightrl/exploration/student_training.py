from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np
import torch

from flightrl.puffer4_edge_schema import EDGE_FRAME_PIXELS
from flightrl.puffer4_edge_training import apply_recurrent_resets
from flightrl.puffer4_edge_training_state import edge_state_dict_sha256

from .policy import CoverageExplorationActor
from .student_metrics import (
    coverage_action_metrics,
    coverage_causal_checks,
    decision_event_mask,
    history_permuted_observation,
    persistence_baseline_metrics,
)
from .student_provenance import coverage_sequence_sha256
from .student_sequence import (
    CoverageSequenceDataset,
    require_coverage_sequence_dataset,
    require_matched_counterfactual_pairs,
)


COVERAGE_TRAINING_REPORT_SCHEMA = "flightrl.coverage.student_training.v1"


@dataclass(frozen=True, slots=True)
class CoverageTrainConfig:
    epochs: int = 20
    learning_rate: float = 2.0e-3
    tbptt_steps: int = 20
    seed: int = 17
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        for name in ("epochs", "tbptt_steps"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"coverage training {name} must be positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("coverage training seed must be nonnegative")
        for name in ("learning_rate", "gradient_clip_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"coverage training {name} must be finite and positive"
                )


class CoverageTrainingRejected(RuntimeError):
    def __init__(self, report: dict) -> None:
        super().__init__("coverage student did not pass the causal camera gate")
        self.report = report


def train_coverage_student(
    train: CoverageSequenceDataset,
    selection: CoverageSequenceDataset,
    config: CoverageTrainConfig,
) -> tuple[CoverageExplorationActor, dict]:
    _require_training_inputs(train, selection, config)
    actor = _fit_actor(train, config, telemetry_only=False)
    telemetry_actor = _fit_actor(train, config, telemetry_only=True)
    clean = evaluate_coverage_sequence(actor, selection)
    permuted = evaluate_coverage_sequence(
        actor, selection, history_permuted=True
    )
    persistence = persistence_baseline_metrics(selection)
    telemetry = evaluate_coverage_sequence(
        telemetry_actor, selection, telemetry_only=True
    )
    checks = coverage_causal_checks(clean, permuted, persistence, telemetry)
    report = _report(
        actor,
        config,
        train=train,
        selection=selection,
        clean=clean,
        permuted=permuted,
        persistence=persistence,
        telemetry=telemetry,
        checks=checks,
    )
    if not all(checks.values()):
        report["status"] = "rejected"
        raise CoverageTrainingRejected(report)
    return actor, report


@torch.no_grad()
def evaluate_coverage_sequence(
    actor: CoverageExplorationActor,
    dataset: CoverageSequenceDataset,
    *,
    history_permuted: bool = False,
    telemetry_only: bool = False,
) -> dict[str, float | int]:
    require_coverage_sequence_dataset(dataset)
    actor.eval()
    state = actor.initial_state(dataset.shape[1])
    predictions = []
    for step in range(dataset.shape[0]):
        state = apply_recurrent_resets(state, dataset.resets[step])
        observation = _observation(
            dataset,
            step,
            history_permuted=history_permuted,
            telemetry_only=telemetry_only,
        )
        action, state = actor.forward_step(observation, state)
        predictions.append(action[:, (0, 3)].cpu())
    return coverage_action_metrics(torch.stack(predictions).numpy(), dataset)


def _fit_actor(
    dataset: CoverageSequenceDataset,
    config: CoverageTrainConfig,
    *,
    telemetry_only: bool,
) -> CoverageExplorationActor:
    torch.manual_seed(config.seed)
    actor = CoverageExplorationActor(hidden_size=48)
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=config.learning_rate, weight_decay=1.0e-5
    )
    target = torch.from_numpy(dataset.teacher_actions)
    weights = _balanced_mode_weights(dataset)
    for _epoch in range(config.epochs):
        actor.train()
        state = actor.initial_state(dataset.shape[1])
        for start in range(0, dataset.shape[0], config.tbptt_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), dtype=torch.float32)
            for step in range(start, start + config.tbptt_steps):
                state = apply_recurrent_resets(state, dataset.resets[step])
                observation = _observation(
                    dataset, step, telemetry_only=telemetry_only
                )
                action, state = actor.forward_step(observation, state)
                squared = torch.square(action[:, (0, 3)] - target[step]).mean(dim=1)
                loss = loss + torch.mean(squared * weights[step])
            loss = loss / config.tbptt_steps
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("coverage training produced a nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            state = state.detach()
    actor.eval()
    return actor


def _observation(
    dataset: CoverageSequenceDataset,
    step: int,
    *,
    history_permuted: bool = False,
    telemetry_only: bool = False,
) -> torch.Tensor:
    value = (
        history_permuted_observation(dataset, step)
        if history_permuted
        else dataset.model_observation(step)
    )
    if telemetry_only:
        value[:, :EDGE_FRAME_PIXELS] = 0.0
    return value


def _balanced_mode_weights(dataset: CoverageSequenceDataset) -> torch.Tensor:
    labels = np.argmin(
        np.square(
            dataset.teacher_actions[..., None, :]
            - np.asarray(((0.5, 0.0), (0.0, 1.0)), dtype=np.float32)
        ).sum(axis=-1),
        axis=-1,
    )
    counts = np.bincount(labels.reshape(-1), minlength=2)
    if np.any(counts == 0):
        raise ValueError("coverage training requires both advance and scan labels")
    per_mode = labels.size / (2.0 * counts)
    return torch.from_numpy(per_mode[labels].astype(np.float32))


def _require_training_inputs(
    train: CoverageSequenceDataset,
    selection: CoverageSequenceDataset,
    config: CoverageTrainConfig,
) -> None:
    require_coverage_sequence_dataset(train)
    require_coverage_sequence_dataset(selection)
    if train.metadata["split"] != "train" or selection.metadata["split"] != "selection":
        raise ValueError("coverage training requires train and selection splits")
    if set(map(int, train.scene_ids)) & set(map(int, selection.scene_ids)):
        raise ValueError("coverage train and selection scene IDs must be disjoint")
    if train.shape[0] % config.tbptt_steps:
        raise ValueError("coverage TBPTT steps must evenly divide the train sequence")
    decision_event_mask(train)
    require_matched_counterfactual_pairs(selection)
    _balanced_mode_weights(train)


def _report(
    actor: CoverageExplorationActor,
    config: CoverageTrainConfig,
    *,
    train: CoverageSequenceDataset,
    selection: CoverageSequenceDataset,
    clean: dict,
    permuted: dict,
    persistence: dict,
    telemetry: dict,
    checks: dict[str, bool],
) -> dict:
    return {
        "schema": COVERAGE_TRAINING_REPORT_SCHEMA,
        "status": "complete",
        "config": asdict(config),
        "datasets": {
            "train": dict(train.metadata)
            | {"sha256": coverage_sequence_sha256(train)},
            "selection": dict(selection.metadata)
            | {"sha256": coverage_sequence_sha256(selection)},
        },
        "selection": clean,
        "selection_history_permuted": permuted,
        "persistence_baseline": persistence,
        "telemetry_only_baseline": telemetry,
        "causal_gate": {"checks": checks, "passed": all(checks.values())},
        "selected_actor_state_sha256": edge_state_dict_sha256(actor.state_dict()),
        "parameter_count": actor.parameter_count,
        "evaluation_scope": "offline_sequence_behavior_cloning",
        "closed_loop_evaluated": False,
        "generalization_authority": False,
        "training_authority": False,
        "deployment_authority": False,
        "flight_authority": False,
    }
