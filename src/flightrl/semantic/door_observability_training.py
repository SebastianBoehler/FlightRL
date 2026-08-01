from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .door_observability import (
    DoorObservabilityNet,
    decode_observability,
    observability_loss,
)


@dataclass(frozen=True, slots=True)
class DoorObservabilityTrainingConfig:
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")


@dataclass(slots=True)
class DoorObservabilityTrainingResult:
    model: DoorObservabilityNet
    final_train_loss: float
    validation_predictions: np.ndarray
    epoch_losses: tuple[float, ...]


def train_door_observability(
    *,
    train_frames: np.ndarray,
    train_labels: np.ndarray,
    validation_frames: np.ndarray,
    validation_labels: np.ndarray,
    config: DoorObservabilityTrainingConfig,
    device: str,
) -> DoorObservabilityTrainingResult:
    _validate_dataset(train_frames, train_labels, "training")
    _validate_dataset(validation_frames, validation_labels, "validation")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = DoorObservabilityNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    frames = torch.from_numpy(train_frames).to(device)
    labels = torch.from_numpy(train_labels).to(device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    epoch_losses: list[float] = []
    for _epoch in range(config.epochs):
        permutation = torch.randperm(
            frames.shape[0],
            generator=generator,
        ).to(device)
        total_loss = 0.0
        batches = 0
        model.train()
        for start in range(0, frames.shape[0], config.batch_size):
            selected = permutation[start : start + config.batch_size]
            loss = observability_loss(model(frames[selected]), labels[selected])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        epoch_losses.append(total_loss / batches)
    model.eval()
    with torch.no_grad():
        validation = torch.from_numpy(validation_frames).to(device)
        predictions = (
            decode_observability(model(validation)).detach().cpu().numpy()
        )
    return DoorObservabilityTrainingResult(
        model=model,
        final_train_loss=epoch_losses[-1],
        validation_predictions=predictions,
        epoch_losses=tuple(epoch_losses),
    )


def _validate_dataset(
    frames: np.ndarray,
    labels: np.ndarray,
    name: str,
) -> None:
    if frames.ndim != 4 or frames.shape[1:] != (1, 48, 64):
        raise ValueError(f"{name} frames must have shape [samples, 1, 48, 64]")
    if labels.shape != (frames.shape[0], 4):
        raise ValueError(f"{name} labels must have shape [samples, 4]")
    if frames.dtype != np.float32 or labels.dtype != np.float32:
        raise ValueError(f"{name} frames and labels must be float32")
    if not np.isfinite(frames).all() or not np.isfinite(labels).all():
        raise ValueError(f"{name} frames and labels must be finite")
