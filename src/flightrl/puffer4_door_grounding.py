from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from flightrl.puffer4_door_policy import (
    DOOR_OBS_DIM,
    DOOR_POLICY_OBS_DIM,
)
from flightrl.puffer4_door_grounding_metrics import (
    calibrate_visibility_threshold,
    grounding_metrics,
)
from flightrl.puffer4_door_mujoco_replay import (
    MUJOCO_SELECTION_ROOM_SEEDS,
    MUJOCO_TRAIN_ROOM_SEEDS,
    collect_default_mujoco_replays,
    evaluate_mujoco_grounder,
    mixed_grounding_selection_score,
    sample_mixed_grounding_batch,
)
from flightrl.puffer4_door_replay import collect_grounding_replay

GROUNDING_TRAIN_SEEDS = (11, 23, 47, 71, 101)
GROUNDING_SELECTION_SEED = 4_001
GROUNDING_TRAIN_APPEARANCE_SEEDS = (2_003, 3_001, 4_003, 5_009, 6_007)
GROUNDING_SELECTION_APPEARANCE_SEED = 8_009
GROUNDING_EVALUATION_APPEARANCE_SEED = 10_007


def door_grounding_labels(observations: torch.Tensor) -> torch.Tensor:
    if observations.shape[-1] != DOOR_OBS_DIM:
        raise ValueError("fixed-door grounding labels use the wrong contract")
    return observations[..., DOOR_POLICY_OBS_DIM + 2 :]


def balanced_visibility_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
    )
    positive = labels > 0.5
    positive_count = torch.sum(positive)
    negative_count = labels.numel() - positive_count
    if positive_count == 0 or negative_count == 0:
        return torch.mean(losses)
    weights = torch.where(
        positive,
        0.5 * labels.numel() / positive_count,
        0.5 * labels.numel() / negative_count,
    )
    return torch.mean(losses * weights)


def grounding_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = logits[:, :4]
    visibility = balanced_visibility_loss(logits[:, 0], labels[:, 0])
    positive = labels[:, 0] > 0.5
    centroid = (
        nn.functional.smooth_l1_loss(
            torch.sigmoid(logits[positive, 1:]),
            labels[positive, 1:],
        )
        if torch.any(positive)
        else logits.sum() * 0.0
    )
    return visibility + 2.0 * centroid, visibility, centroid


def _grounding_stream(vec, torch_pufferl) -> tuple:
    observations = torch_pufferl._cpu_tensor(
        vec.obs_ptr,
        (vec.total_agents, vec.obs_size),
        torch.float32,
    )
    return vec, observations


def _create_grounding_vec(
    args: dict,
    torch_pufferl,
    seed: int,
    appearance_seed: int,
):
    stream_args = deepcopy(args)
    stream_args["env"]["seed"] = seed
    stream_args["env"]["appearance_seed"] = appearance_seed
    return torch_pufferl._C.create_vec(stream_args, torch_pufferl._C.gpu)


def train_door_grounder(
    policy,
    vec,
    args,
    torch_pufferl,
    *,
    updates: int,
    learning_rate: float,
    training_seeds: tuple[int, ...] = GROUNDING_TRAIN_SEEDS,
    training_appearance_seeds: tuple[int, ...] = (
        GROUNDING_TRAIN_APPEARANCE_SEEDS
    ),
    selection_seed: int = GROUNDING_SELECTION_SEED,
    selection_appearance_seed: int = GROUNDING_SELECTION_APPEARANCE_SEED,
    selection_batches: int = 16,
    log_interval: int = 64,
) -> dict:
    if not training_seeds:
        raise ValueError("fixed-door grounder requires at least one training seed")
    if len(training_appearance_seeds) != len(training_seeds):
        raise ValueError("native training seeds and appearance seeds must align")
    owned_vecs = []
    streams = []
    for seed, appearance_seed in zip(
        training_seeds,
        training_appearance_seeds,
        strict=True,
    ):
        matches_primary = (
            seed == int(args["env"]["seed"])
            and appearance_seed == int(args["env"]["appearance_seed"])
        )
        stream_vec = vec if matches_primary else _create_grounding_vec(
            args,
            torch_pufferl,
            seed,
            appearance_seed,
        )
        if stream_vec is not vec:
            owned_vecs.append(stream_vec)
        streams.append(_grounding_stream(stream_vec, torch_pufferl))
    parameters = tuple(policy.encoder.grounder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    initial_loss = 0.0
    final_loss = 0.0
    metrics = {}
    best_state = deepcopy(policy.encoder.grounder.state_dict())
    best_update = 0
    best_metrics = evaluate_door_grounder(
        policy,
        args,
        torch_pufferl,
        batches=selection_batches,
        seed=selection_seed,
        appearance_seed=selection_appearance_seed,
        agents=min(vec.total_agents, 256),
        calibrate=True,
    )
    mujoco_training, mujoco_selection = collect_default_mujoco_replays()
    best_mujoco_metrics = evaluate_mujoco_grounder(
        policy.encoder.grounder,
        mujoco_selection,
        visibility_threshold=best_metrics["visibility_threshold"],
    )
    best_score = mixed_grounding_selection_score(
        best_metrics,
        best_mujoco_metrics,
    )
    replay_batches = max(8, min(64, updates // 4))
    try:
        replay_frames, replay_labels = collect_grounding_replay(
            streams,
            batches=replay_batches,
        )
    finally:
        for owned in owned_vecs:
            owned.close()
    generator = torch.Generator(device="cpu").manual_seed(1_701)
    batch_size = min(1_024, replay_frames.shape[0])
    try:
        for update in range(updates):
            frames, labels = sample_mixed_grounding_batch(
                replay_frames,
                replay_labels,
                mujoco_training,
                batch_size=batch_size,
                generator=generator,
            )
            logits = policy.encoder.grounder(frames)
            loss, _, _ = grounding_loss(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if update == 0:
                initial_loss = float(loss.detach())
            final_loss = float(loss.detach())
            should_log = (
                update == 0
                or update + 1 == updates
                or (update + 1) % log_interval == 0
            )
            if not should_log:
                continue
            metrics = grounding_metrics(logits, labels)
            selection = evaluate_door_grounder(
                policy,
                args,
                torch_pufferl,
                batches=selection_batches,
                seed=selection_seed,
                appearance_seed=selection_appearance_seed,
                agents=min(vec.total_agents, 256),
                calibrate=True,
            )
            mujoco_metrics = evaluate_mujoco_grounder(
                policy.encoder.grounder,
                mujoco_selection,
                visibility_threshold=selection["visibility_threshold"],
            )
            score = mixed_grounding_selection_score(
                selection,
                mujoco_metrics,
            )
            if score > best_score:
                best_score = score
                best_update = update + 1
                best_metrics = selection
                best_mujoco_metrics = mujoco_metrics
                best_state = deepcopy(policy.encoder.grounder.state_dict())
            print(
                f"grounder={update + 1}/{updates} loss={final_loss:.6f} "
                f"train={metrics} native={selection} "
                f"mujoco={mujoco_metrics}",
                flush=True,
            )
    finally:
        del replay_frames, replay_labels
    policy.encoder.grounder.load_state_dict(best_state)
    return {
        "updates": updates,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "last_batch": metrics,
        "training_seeds": list(training_seeds),
        "training_appearance_seeds": list(training_appearance_seeds),
        "training_samples": len(streams) * replay_batches * vec.total_agents,
        "mujoco_training_room_seeds": list(MUJOCO_TRAIN_ROOM_SEEDS),
        "mujoco_training_samples": mujoco_training.sample_count,
        "selection_seed": selection_seed,
        "selection_appearance_seed": selection_appearance_seed,
        "mujoco_selection_room_seeds": list(MUJOCO_SELECTION_ROOM_SEEDS),
        "mujoco_selection_samples": mujoco_selection.sample_count,
        "best_selection_update": best_update,
        "best_selection_metrics": best_metrics,
        "best_mujoco_selection_metrics": best_mujoco_metrics,
        "best_mixed_selection_score": best_score,
    }


@torch.no_grad()
def evaluate_door_grounder(
    policy,
    args: dict,
    torch_pufferl,
    *,
    batches: int,
    seed: int,
    agents: int,
    appearance_seed: int | None = None,
    visibility_threshold: float = 0.5,
    calibrate: bool = False,
) -> dict[str, float]:
    eval_args = deepcopy(args)
    eval_args["env"]["seed"] = seed
    if appearance_seed is not None:
        eval_args["env"]["appearance_seed"] = appearance_seed
    eval_args["vec"]["total_agents"] = agents
    vec = torch_pufferl._C.create_vec(eval_args, torch_pufferl._C.gpu)
    observations = torch_pufferl._cpu_tensor(
        vec.obs_ptr,
        (vec.total_agents, vec.obs_size),
        torch.float32,
    )
    all_logits = []
    all_labels = []
    for _ in range(batches):
        vec.reset()
        all_logits.append(
            policy.encoder.predict_grounding(observations).cpu().clone()
        )
        all_labels.append(door_grounding_labels(observations).cpu().clone())
    vec.close()
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    if calibrate:
        visibility_threshold = calibrate_visibility_threshold(logits, labels)
    return grounding_metrics(
        logits,
        labels,
        visibility_threshold=visibility_threshold,
    )
