from __future__ import annotations

import torch

from flightrl.puffer4_door_policy import (
    DOOR_OBS_DIM,
    DOOR_PIXELS,
    DOOR_POLICY_OBS_DIM,
)
from flightrl.puffer4_door_observation import DOOR_SENSOR_DIM
from flightrl.puffer4_door_grounding import (
    door_grounding_labels,
    grounding_loss,
    grounding_metrics,
)


BOOTSTRAP_HORIZON = 64


def door_teacher_actions(observations: torch.Tensor) -> torch.Tensor:
    if observations.shape[1] != DOOR_OBS_DIM:
        raise ValueError("fixed-door bootstrap received the wrong observation contract")
    start = DOOR_POLICY_OBS_DIM
    return observations[:, start : start + 2].clone()


def door_imitation_sample_weights(
    observations: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if observations.shape[0] != targets.shape[0] or targets.shape[-1] != 2:
        raise ValueError("door imitation observations and targets must align")
    phase_offset = 3 * DOOR_PIXELS + DOOR_SENSOR_DIM
    search = observations[:, phase_offset] > 0.5
    turning = (targets[:, 0] < 0.05) & ~search
    return 1.0 + 4.0 * search.float() + 2.0 * turning.float()


def initialize_door_observability(policy, checkpoint: dict) -> None:
    policy.encoder.load_observability_checkpoint(checkpoint)


def freeze_door_grounder(policy) -> None:
    for parameter in policy.encoder.grounder.parameters():
        parameter.requires_grad_(False)


def load_compatible_policy_state(policy, state: dict | None) -> dict[str, int]:
    if state is None:
        return {"loaded_tensors": 0, "skipped_tensors": 0}
    current = policy.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    policy.load_state_dict(compatible, strict=False)
    return {
        "loaded_tensors": len(compatible),
        "skipped_tensors": len(state) - len(compatible),
    }


def _forward_sequence(
    policy,
    observations: torch.Tensor,
    terminals: torch.Tensor,
    initial_state: tuple[torch.Tensor, ...],
):
    batch, horizon = observations.shape[:2]
    encoded, grounding = policy.encoder.encode_with_grounding(
        observations.reshape(batch * horizon, -1)
    )
    encoded = encoded.reshape(batch, horizon, -1)
    hidden, next_state = _forward_min_gru_sequence(
        policy.network,
        encoded,
        terminals,
        initial_state,
    )
    hidden = hidden.reshape(batch * horizon, -1)
    distribution, _ = policy.decoder(hidden)
    return distribution, grounding, next_state


def _forward_min_gru_sequence(
    network,
    hidden: torch.Tensor,
    terminals: torch.Tensor,
    initial_state: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    required = ("layers", "_g", "_log_g", "_highway", "_heinsen_scan")
    if not all(hasattr(network, name) for name in required):
        raise TypeError("door bootstrap requires the configured MinGRU network")
    recurrent_state = initial_state[0]
    if recurrent_state.shape[0] != len(network.layers):
        raise ValueError("MinGRU initial state does not match its layer count")
    reset_log = torch.zeros_like(terminals)
    reset_log[:, 1:] = -30.0 * terminals[:, :-1]
    next_states = []
    for layer_index, layer in enumerate(network.layers):
        candidate, gate, projection = layer(hidden).chunk(3, dim=-1)
        log_coefficients = (
            -torch.nn.functional.softplus(gate)
            + reset_log.unsqueeze(-1)
        )
        log_values = (
            -torch.nn.functional.softplus(-gate)
            + network._log_g(candidate)
        )
        state = recurrent_state[layer_index]
        state_log = torch.where(
            state > 0.0,
            torch.log(state),
            state.new_full((), float("-inf")),
        )
        first_value = torch.logaddexp(
            log_values[:, 0],
            log_coefficients[:, 0] + state_log,
        )
        log_values = torch.cat(
            (first_value[:, None], log_values[:, 1:]),
            dim=1,
        )
        output = network._heinsen_scan(log_coefficients, log_values)
        next_states.append(
            output[:, -1] * (1.0 - terminals[:, -1, None])
        )
        hidden = network._highway(hidden, output, projection)
    return hidden, (torch.stack(next_states),)


def bootstrap_door_policy(
    policy,
    vec,
    torch_pufferl,
    *,
    updates: int,
    learning_rate: float,
    max_policy_rollin: float,
) -> dict[str, float | int]:
    if updates <= 0:
        return {"updates": 0}
    observations = torch_pufferl._cpu_tensor(
        vec.obs_ptr,
        (vec.total_agents, vec.obs_size),
        torch.float32,
    )
    terminals = torch_pufferl._cpu_tensor(
        vec.terminals_ptr,
        (vec.total_agents,),
        torch.float32,
    )
    parameters = tuple(item for item in policy.parameters() if item.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    vec.reset()
    state = policy.initial_state(vec.total_agents, device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(17)
    initial_loss = 0.0
    final_loss = 0.0
    final_action_loss = 0.0
    final_visibility_loss = 0.0
    for update in range(updates):
        initial_state = tuple(item.detach().clone() for item in state)
        policy_fraction = max_policy_rollin * min(
            1.0,
            (update + 1) / max(1, updates // 2),
        )
        sequence_observations = []
        sequence_targets = []
        sequence_terminals = []
        for _ in range(BOOTSTRAP_HORIZON):
            targets = door_teacher_actions(observations)
            sequence_observations.append(observations.clone())
            sequence_targets.append(targets)
            with torch.no_grad():
                if max_policy_rollin > 0.0:
                    distribution, _, state = policy.forward_eval(
                        observations,
                        state,
                    )
                    predicted = distribution.mean.clamp(-1.0, 1.0)
                    predicted[:, 0].clamp_(0.0, 1.0)
                    use_policy = (
                        torch.rand(vec.total_agents, generator=generator)
                        < policy_fraction
                    )
                    rollout = torch.where(
                        use_policy[:, None],
                        predicted,
                        targets,
                    ).contiguous()
                else:
                    rollout = targets.contiguous()
            vec.cpu_step(rollout.data_ptr())
            sequence_terminals.append(terminals.clone())
            if max_policy_rollin > 0.0:
                alive = (1.0 - terminals).view(1, -1, 1)
                state = tuple(item * alive for item in state)
        batch = torch.stack(sequence_observations, dim=1)
        targets = torch.stack(sequence_targets, dim=1).reshape(-1, 2)
        terminal_batch = torch.stack(sequence_terminals, dim=1)
        distribution, grounding, training_state = _forward_sequence(
            policy,
            batch,
            terminal_batch,
            initial_state,
        )
        flat_observations = batch.reshape(-1, batch.shape[-1])
        sample_weight = door_imitation_sample_weights(
            flat_observations,
            targets,
        )
        search = sample_weight == 5.0
        action_weight = targets.new_tensor((1.0, 3.0))
        squared = (distribution.mean - targets) ** 2 * action_weight
        action_loss = torch.sum(squared * sample_weight[:, None]) / (
            torch.sum(sample_weight) * torch.sum(action_weight)
        )
        labels = door_grounding_labels(batch).reshape(-1, 4)
        ground_loss, visibility_loss, centroid_loss = grounding_loss(
            grounding,
            labels,
        )
        loss = action_loss + ground_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if max_policy_rollin <= 0.0:
            state = tuple(item.detach() for item in training_state)
        if update == 0:
            initial_loss = float(loss.detach())
        final_loss = float(loss.detach())
        final_action_loss = float(action_loss.detach())
        final_visibility_loss = float(visibility_loss.detach())
        if update == 0 or update + 1 == updates or (update + 1) % 16 == 0:
            metrics = grounding_metrics(grounding, labels)
            print(
                f"bootstrap={update + 1}/{updates} loss={final_loss:.6f} "
                f"action_loss={final_action_loss:.6f} "
                f"visibility_bce={final_visibility_loss:.6f} "
                f"centroid_loss={float(centroid_loss.detach()):.6f} "
                f"visibility_recall={metrics['visibility_recall']:.3f} "
                f"visibility_fpr={metrics['visibility_false_positive_rate']:.3f} "
                f"search_fraction={float(search.float().mean()):.3f}",
                flush=True,
            )
    return {
        "updates": updates,
        "horizon": BOOTSTRAP_HORIZON,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_action_loss": final_action_loss,
        "final_visibility_loss": final_visibility_loss,
        "final_policy_rollin_fraction": policy_fraction,
    }
