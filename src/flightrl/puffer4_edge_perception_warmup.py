from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

from flightrl.puffer4_edge_contract import (
    EDGE_FRAME_PIXELS,
    EDGE_HEIGHT,
    EDGE_MISSION_TOKEN_COUNT,
    EDGE_WIDTH,
)
from flightrl.puffer4_edge_sequence import (
    EdgeSequenceDataset,
    require_edge_sequence_structure,
)
from flightrl.puffer4_edge_training_data import (
    edge_grounding_losses,
    edge_sequence_loss_weights,
)
from flightrl.puffer4_edge_training_state import edge_state_dict_sha256


PERCEPTION_MODULE_NAMES = (
    "visual",
    "grounding_target_gate",
    "grounding_head",
)
WARMUP_REPORT_FIELDS = {
    "history",
    "selected_epoch",
    "selected_selection_metrics",
    "selected_state_sha256",
    "frozen_parameter_names",
    "sampling",
}


def edge_perception_batch_orders(
    sample_count: int,
    batch_size: int,
    epochs: int,
    *,
    seed: int,
) -> Iterator[tuple[tuple[int, ...], ...]]:
    _require_positive_int(sample_count, "sample count")
    _require_positive_int(batch_size, "batch size")
    _require_positive_int(epochs, "epochs")
    if type(seed) is not int or seed < 0:
        raise ValueError("edge perception seed must be nonnegative")
    if sample_count < batch_size or sample_count % batch_size != 0:
        raise ValueError(
            "edge perception samples must divide evenly into full batches"
        )
    generator = np.random.Generator(np.random.PCG64(seed))
    for _epoch in range(epochs):
        order = generator.permutation(sample_count)
        yield tuple(
            tuple(int(value) for value in order[start : start + batch_size])
            for start in range(0, sample_count, batch_size)
        )


def require_edge_perception_warmup_batches(dataset, config) -> None:
    require_edge_sequence_structure(dataset)
    next(
        edge_perception_batch_orders(
            int(np.prod(dataset.shape)),
            config.warmup_batch_size,
            config.warmup_epochs,
            seed=config.seed,
        )
    )


def edge_perception_parameter_names(actor) -> tuple[str, ...]:
    prefixes = tuple(f"{name}." for name in PERCEPTION_MODULE_NAMES)
    names = tuple(
        name for name, _parameter in actor.named_parameters()
        if name.startswith(prefixes)
    )
    if not names:
        raise ValueError("edge actor has no perception parameters")
    return names


def edge_perception_state_dict(actor) -> dict[str, torch.Tensor]:
    names = set(edge_perception_parameter_names(actor))
    return {
        name: value
        for name, value in actor.state_dict().items()
        if name in names
    }


def edge_control_state_dict(actor) -> dict[str, torch.Tensor]:
    names = set(edge_perception_parameter_names(actor))
    return {
        name: value
        for name, value in actor.state_dict().items()
        if name not in names
    }


def edge_perception_state_sha256(actor) -> str:
    return edge_state_dict_sha256(edge_perception_state_dict(actor))


def warmup_edge_perception(
    actor,
    train: EdgeSequenceDataset,
    selection: EdgeSequenceDataset,
    config,
) -> dict:
    require_edge_perception_warmup_batches(train, config)
    require_edge_sequence_structure(selection)
    control_digest = edge_state_dict_sha256(edge_control_state_dict(actor))
    parameters = [
        parameter
        for name, parameter in actor.named_parameters()
        if name in set(edge_perception_parameter_names(actor))
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.perception_learning_rate,
        weight_decay=1.0e-5,
    )
    weights = edge_sequence_loss_weights(train)
    history = []
    selected_epoch = 0
    selected_loss = float("inf")
    selected_state = None
    orders = edge_perception_batch_orders(
        int(np.prod(train.shape)),
        config.warmup_batch_size,
        config.warmup_epochs,
        seed=config.seed,
    )
    for epoch, batches in enumerate(orders, 1):
        actor.train()
        _train_perception_epoch(
            actor,
            optimizer,
            train,
            weights,
            config,
            batches,
        )
        metrics = evaluate_edge_grounding(actor, selection, config)
        history.append({"epoch": epoch, "selection": metrics})
        if metrics["grounding_loss"] < selected_loss:
            selected_epoch = epoch
            selected_loss = metrics["grounding_loss"]
            selected_state = {
                name: value.detach().cpu().clone()
                for name, value in edge_perception_state_dict(actor).items()
            }
    if selected_state is None:
        raise RuntimeError("edge perception warmup did not select a state")
    state = actor.state_dict()
    state.update(selected_state)
    actor.load_state_dict(state, strict=True)
    if edge_state_dict_sha256(edge_control_state_dict(actor)) != control_digest:
        raise RuntimeError("edge perception warmup changed control parameters")
    frozen_names = edge_perception_parameter_names(actor)
    for name, parameter in actor.named_parameters():
        if name in set(frozen_names):
            parameter.requires_grad_(False)
    actor.eval()
    return {
        "history": history,
        "selected_epoch": selected_epoch,
        "selected_selection_metrics": dict(history[selected_epoch - 1]["selection"]),
        "selected_state_sha256": edge_perception_state_sha256(actor),
        "frozen_parameter_names": list(frozen_names),
        "sampling": {
            "rng": "numpy.PCG64",
            "flattening": "step_major_agent_minor",
            "order": "full_permutation_without_replacement",
            "samples_per_epoch": int(np.prod(train.shape)),
            "batches_per_epoch": int(np.prod(train.shape))
            // config.warmup_batch_size,
        },
    }


@torch.no_grad()
def evaluate_edge_grounding(actor, dataset, config) -> dict[str, float]:
    require_edge_sequence_structure(dataset)
    weights = edge_sequence_loss_weights(dataset)
    totals = {"visibility": 0.0, "box": 0.0, "grounding": 0.0}
    sample_count = int(np.prod(dataset.shape))
    actor.eval()
    for start in range(0, sample_count, config.warmup_batch_size):
        indices = np.arange(
            start,
            min(start + config.warmup_batch_size, sample_count),
            dtype=np.int64,
        )
        grounding, visibility_logit, target = _perception_batch(
            actor, dataset, indices
        )
        losses = edge_grounding_losses(
            grounding,
            visibility_logit,
            target,
            weights.visibility.reshape(-1)[indices],
            weights.box.reshape(-1)[indices],
            config,
        )
        batch_size = len(indices)
        for name, value in losses.items():
            totals[name] += float(value) * batch_size
    return {
        f"{name}_loss": value / sample_count
        for name, value in totals.items()
    }


def _train_perception_epoch(
    actor,
    optimizer,
    dataset,
    weights,
    config,
    batches,
) -> None:
    flat_visibility = weights.visibility.reshape(-1)
    flat_box = weights.box.reshape(-1)
    for batch in batches:
        indices = np.asarray(batch, dtype=np.int64)
        grounding, visibility_logit, target = _perception_batch(
            actor, dataset, indices
        )
        losses = edge_grounding_losses(
            grounding,
            visibility_logit,
            target,
            flat_visibility[indices],
            flat_box[indices],
            config,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["grounding"].backward()
        torch.nn.utils.clip_grad_norm_(
            tuple(parameter for group in optimizer.param_groups for parameter in group["params"]),
            config.gradient_clip_norm,
        )
        optimizer.step()


def _perception_batch(actor, dataset, flat_indices):
    agents = dataset.shape[1]
    steps = flat_indices // agents
    agent_indices = flat_indices % agents
    packed = torch.from_numpy(dataset.packed_frames[steps, agent_indices])
    pixels = torch.empty(len(flat_indices), EDGE_FRAME_PIXELS, dtype=torch.float32)
    pixels[:, 0::2] = (packed >> 4).to(torch.float32) / 15.0
    pixels[:, 1::2] = (packed & 0x0F).to(torch.float32) / 15.0
    frame = pixels.reshape(-1, 1, EDGE_HEIGHT, EDGE_WIDTH)
    target_ids = torch.from_numpy(dataset.target_ids[steps, agent_indices]).long()
    mission = torch.nn.functional.one_hot(
        target_ids, EDGE_MISSION_TOKEN_COUNT
    ).to(torch.float32)
    target = torch.from_numpy(dataset.grounding[steps, agent_indices])
    visual = actor.visual(frame)
    grounding, visibility_logit = actor._grounding_with_logit(visual, mission)
    return grounding, visibility_logit, target


def _require_positive_int(value: object, label: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"edge perception {label} must be a positive integer")
