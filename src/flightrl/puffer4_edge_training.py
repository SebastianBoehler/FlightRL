from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch

from flightrl.puffer4_edge_policy import EdgeNavigationActor
from flightrl.puffer4_edge_native_build import (
    require_matching_edge_native_build_fingerprints,
)
from flightrl.puffer4_edge_sequence import (
    EdgeSequenceDataset,
    require_disjoint_edge_datasets,
    require_edge_sequence_dataset,
    require_matching_edge_dataset_environments,
)
from flightrl.puffer4_edge_training_data import (
    accumulate_losses as _accumulate,
    average_losses as _averages,
    balanced_visibility_loss as balanced_visibility_loss,
    edge_sequence_loss_weights,
    edge_step_losses as _losses,
    edge_training_baseline_values,
    empty_loss_totals as _empty_totals,
)
from flightrl.puffer4_edge_training_report import (
    EDGE_LOSS_CONTRACT as EDGE_LOSS_CONTRACT,
    EDGE_SELECTION_RULE as EDGE_SELECTION_RULE,
    EDGE_TRAINING_REPORT_SCHEMA as EDGE_TRAINING_REPORT_SCHEMA,
    EDGE_WEIGHTING_CONTRACT as EDGE_WEIGHTING_CONTRACT,
    edge_training_report,
)
from flightrl.puffer4_edge_training_selection import (
    cyclic_selection_frame_ablation,
    edge_baseline_checks,
)


def edge_training_baselines(
    dataset: EdgeSequenceDataset,
    config: "EdgeTrainConfig",
) -> dict[str, dict[str, float | list[float]]]:
    return edge_training_baseline_values(
        dataset,
        visibility_loss_weight=config.visibility_loss_weight,
        box_loss_weight=config.box_loss_weight,
    )


@dataclass(frozen=True, slots=True)
class EdgeTrainConfig:
    epochs: int = 8
    learning_rate: float = 2.0e-3
    tbptt_steps: int = 40
    seed: int = 17
    gradient_clip_norm: float = 1.0
    visibility_loss_weight: float = 0.30
    box_loss_weight: float = 0.20

    def __post_init__(self) -> None:
        for name in ("epochs", "tbptt_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"edge training {name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("edge training seed must be nonnegative")
        for name in (
            "learning_rate",
            "gradient_clip_norm",
            "visibility_loss_weight",
            "box_loss_weight",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(value) or value <= 0.0:
                raise ValueError(f"edge training {name} must be finite and positive")


class EdgeTrainingRejected(RuntimeError):
    def __init__(self, report: dict) -> None:
        super().__init__("edge training did not beat selection baselines")
        self.report = report


def train_edge_student(
    train: EdgeSequenceDataset,
    selection: EdgeSequenceDataset,
    config: EdgeTrainConfig,
) -> tuple[EdgeNavigationActor, dict]:
    require_disjoint_edge_datasets(train, selection)
    require_matching_edge_native_build_fingerprints(
        train.metadata["native_build_fingerprint"],
        selection.metadata["native_build_fingerprint"],
    )
    require_matching_edge_dataset_environments(train, selection)
    if selection.shape[1] < 2:
        raise ValueError("visual ablation requires at least two selection agents")
    require_even_edge_tbptt_chunks(train, config)
    torch.manual_seed(config.seed)
    actor = EdgeNavigationActor(hidden_size=48)
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=config.learning_rate,
        weight_decay=1.0e-5,
    )
    train_weights = edge_sequence_loss_weights(train)
    selection_weights = edge_sequence_loss_weights(selection)
    baselines = edge_training_baselines(selection, config)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    fallback_loss = float("inf")
    fallback_state: dict[str, torch.Tensor] | None = None
    fallback_record: dict | None = None
    history = []
    for epoch in range(1, config.epochs + 1):
        actor.train()
        train_metrics = _train_epoch(actor, optimizer, train, config, train_weights)
        actor.eval()
        selection_metrics = _evaluate_edge_sequence_loss(
            actor, selection, config, selection_weights
        )
        ablated_metrics = _evaluate_edge_sequence_loss(
            actor, selection, config, selection_weights, visual_ablation=True
        )
        checks = edge_baseline_checks(selection_metrics, ablated_metrics, baselines)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "selection": selection_metrics,
            "selection_visual_ablation": ablated_metrics,
            "baseline_checks": checks,
        }
        history.append(record)
        state = {
            name: value.detach().cpu().clone()
            for name, value in actor.state_dict().items()
        }
        if selection_metrics["selection_score"] < fallback_loss:
            fallback_loss = selection_metrics["selection_score"]
            fallback_state = state
            fallback_record = record
        if all(checks.values()) and selection_metrics["selection_score"] < best_loss:
            best_loss = selection_metrics["selection_score"]
            best_epoch = epoch
            best_state = state
    if best_state is None:
        assert fallback_state is not None and fallback_record is not None
        actor.load_state_dict(fallback_state, strict=True)
        actor.eval()
        raise EdgeTrainingRejected(
            edge_training_report(
                actor,
                config,
                history,
                baselines,
                status="rejected",
                selected_record=fallback_record,
            )
        )
    actor.load_state_dict(best_state, strict=True)
    actor.eval()
    report = edge_training_report(
        actor,
        config,
        history,
        baselines,
        status="complete",
        selected_record=history[best_epoch - 1],
    )
    return actor, report


def require_even_edge_tbptt_chunks(
    dataset: EdgeSequenceDataset,
    config: EdgeTrainConfig,
) -> None:
    if dataset.shape[0] % config.tbptt_steps != 0:
        raise ValueError("edge training steps must divide evenly into TBPTT chunks")


@torch.no_grad()
def evaluate_edge_sequence_loss(
    actor: EdgeNavigationActor,
    dataset: EdgeSequenceDataset,
    config: EdgeTrainConfig,
) -> dict[str, float]:
    require_edge_sequence_dataset(dataset)
    return _evaluate_edge_sequence_loss(
        actor,
        dataset,
        config,
        edge_sequence_loss_weights(dataset),
    )


@torch.no_grad()
def evaluate_edge_visual_ablation_loss(
    actor: EdgeNavigationActor,
    dataset: EdgeSequenceDataset,
    config: EdgeTrainConfig,
) -> dict[str, float]:
    require_edge_sequence_dataset(dataset)
    if dataset.shape[1] < 2:
        raise ValueError("visual ablation requires at least two selection agents")
    return _evaluate_edge_sequence_loss(
        actor,
        dataset,
        config,
        edge_sequence_loss_weights(dataset),
        visual_ablation=True,
    )


def _evaluate_edge_sequence_loss(
    actor, dataset, config, weights, *, visual_ablation=False
) -> dict[str, float]:
    state = actor.initial_state(dataset.shape[1])
    totals = _empty_totals()
    for step in range(dataset.shape[0]):
        state = apply_recurrent_resets(state, dataset.resets[step])
        observation = dataset.model_observation(step)
        if visual_ablation:
            observation = cyclic_selection_frame_ablation(observation)
        action, grounding, visibility_logit, state = actor.forward_training_step(
            observation, state
        )
        losses = _losses(
            action, grounding, visibility_logit, dataset, weights, step, config
        )
        _accumulate(totals, losses)
    return _averages(totals, dataset.shape[0])


def apply_recurrent_resets(
    state: torch.Tensor,
    reset: torch.Tensor | object,
) -> torch.Tensor:
    flags = torch.as_tensor(reset, dtype=torch.bool, device=state.device)
    if flags.shape != (state.shape[0],):
        raise ValueError("edge recurrent reset flags do not match the batch")
    return torch.where(flags.unsqueeze(1), torch.zeros_like(state), state)


def _train_epoch(actor, optimizer, dataset, config, weights) -> dict[str, float]:
    state = actor.initial_state(dataset.shape[1])
    totals = _empty_totals()
    for start in range(0, dataset.shape[0], config.tbptt_steps):
        optimizer.zero_grad(set_to_none=True)
        end = min(start + config.tbptt_steps, dataset.shape[0])
        chunk_loss = torch.zeros((), dtype=torch.float32)
        for step in range(start, end):
            state = apply_recurrent_resets(state, dataset.resets[step])
            action, grounding, visibility_logit, state = actor.forward_training_step(
                dataset.model_observation(step),
                state,
            )
            losses = _losses(
                action, grounding, visibility_logit, dataset, weights, step, config
            )
            chunk_loss = chunk_loss + losses["total"]
            _accumulate(totals, losses)
        chunk_loss = chunk_loss / (end - start)
        if not bool(torch.isfinite(chunk_loss)):
            raise RuntimeError("edge training produced a nonfinite loss")
        chunk_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), config.gradient_clip_norm)
        optimizer.step()
        state = state.detach()
    return _averages(totals, dataset.shape[0])
