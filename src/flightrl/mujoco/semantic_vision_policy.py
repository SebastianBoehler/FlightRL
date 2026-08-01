from __future__ import annotations

import torch
from torch import nn

from flightrl.mujoco.semantic_action_projection import project_semantic_actions
from flightrl.mujoco.semantic_observation import (
    GROUNDING_CONFIDENCE_INDEX,
    GROUNDING_HORIZONTAL_ERROR_INDEX,
)
from flightrl.mujoco.semantic_safety_encoder import (
    RecurrentSafetyModel,
    RecurrentVisualSafetyModel,
    VisualSafetyModel,
)
from flightrl.mujoco.semantic_vision_encoder import SemanticVisionEncoder
from flightrl.policy import DefaultDecoder, MinGRU

class SemanticVisionPolicy(nn.Module):
    def __init__(
        self,
        env,
        hidden_size: int = 128,
        num_layers: int = 1,
        *,
        shared_visual_safety: bool = False,
        recurrent_safety: bool = True,
        recurrent_visual_safety: bool = False,
    ) -> None:
        super().__init__()
        observation_size = int(env.single_observation_space.shape[0])
        action_size = int(env.single_action_space.shape[0])
        self.hidden_size = int(hidden_size)
        self.encoder = SemanticVisionEncoder(
            observation_size,
            self.hidden_size,
            vision_config=env.vision_config,
            memory_config=env.memory_config,
        )
        self.network = MinGRU(self.hidden_size, num_layers=num_layers)
        self.decoder = DefaultDecoder(action_size, self.hidden_size)
        self.action_mode = getattr(env, "semantic_action_mode", "target_gated")
        use_recurrent_safety = (
            recurrent_safety
            and not shared_visual_safety
            and not recurrent_visual_safety
        )
        self.recurrent_visual_safety = (
            RecurrentVisualSafetyModel(self.encoder.layout)
            if self.action_mode == "active_exploration"
            and recurrent_visual_safety
            else None
        )
        self.recurrent_safety = (
            RecurrentSafetyModel(self.hidden_size)
            if self.action_mode == "active_exploration" and use_recurrent_safety
            else None
        )
        self.visual_safety = (
            VisualSafetyModel(self.encoder.layout)
            if (
                self.action_mode == "active_exploration"
                and self.recurrent_visual_safety is None
                and not use_recurrent_safety
                and not shared_visual_safety
            )
            else None
        )
        self.clearance_head = (
            nn.Linear(self.encoder.vision_feature_dim, 1)
            if self.action_mode == "active_exploration" and shared_visual_safety
            else None
        )
        self.collision_risk_head = (
            nn.Linear(self.encoder.vision_feature_dim, 1)
            if self.clearance_head is not None
            else None
        )

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor]:
        state = self.network.initial_state(batch_size, device)
        if self.recurrent_visual_safety is not None:
            state += self.recurrent_visual_safety.initial_state(
                batch_size,
                device,
            )
        return state

    def forward_eval(
        self,
        observations: torch.Tensor,
        state: tuple[torch.Tensor] | None = None,
    ):
        distribution, values, next_state, _clearance, _risk = (
            self.forward_eval_with_aux(observations, state)
        )
        return distribution, values, next_state

    def forward_eval_with_aux(
        self,
        observations: torch.Tensor,
        state: tuple[torch.Tensor] | None = None,
    ):
        batch = observations.shape[0]
        hidden = self.encoder(observations)
        navigation_state, safety_state = self._split_state(state)
        hidden, next_navigation_state = self.network.forward_eval(
            hidden,
            self.network.initial_state(batch, observations.device)
            if navigation_state is None
            else navigation_state,
        )
        distribution, values = self.decoder(hidden)
        if self.recurrent_visual_safety is None:
            clearance_m, collision_risk = self._safety_estimates(
                observations,
                hidden,
            )
            next_state = next_navigation_state
        else:
            if safety_state is None:
                safety_state = self.recurrent_visual_safety.initial_state(
                    batch,
                    observations.device,
                )
            clearance_m, collision_risk, next_safety_state = (
                self.recurrent_visual_safety.forward_eval(
                    observations,
                    safety_state,
                )
            )
            next_state = next_navigation_state + next_safety_state
        return (
            self._gate_actions(
                distribution,
                observations,
                clearance_m,
                collision_risk,
            ),
            values,
            next_state,
            clearance_m,
            collision_risk,
        )

    def forward(self, observations: torch.Tensor, state=None):
        del state
        if observations.ndim == 2:
            distribution, values, _ = self.forward_eval(observations)
            return distribution, values
        distribution, values, _state, _clearance, _risk = self.forward_train_with_aux(
            observations
        )
        return distribution, values

    def forward_train_recurrent(
        self,
        observations: torch.Tensor,
        state: tuple[torch.Tensor],
        terminals: torch.Tensor,
    ):
        distribution, values, next_state, _clearance, _risk = (
            self.forward_train_with_aux(
                observations,
                state=state,
                terminals=terminals,
            )
        )
        return distribution, values, next_state

    def forward_train_with_aux(
        self,
        observations: torch.Tensor,
        state: tuple[torch.Tensor] | None = None,
        terminals: torch.Tensor | None = None,
    ):
        if observations.ndim != 3:
            raise ValueError(
                "semantic policy expects [batch, obs] or [batch, time, obs]"
            )
        batch, horizon, observation_size = observations.shape
        flat_observations = observations.reshape(batch * horizon, observation_size)
        hidden = self.encoder(flat_observations)
        sequence = hidden.reshape(batch, horizon, -1)
        navigation_state, safety_state = self._split_state(state)
        if navigation_state is None:
            hidden = self.network.forward_train(sequence)
            next_state = None
        elif terminals is not None:
            hidden, next_state = self.network.forward_train_stateful_masked(
                sequence,
                navigation_state,
                terminals,
            )
        else:
            hidden, next_state = self.network.forward_train_stateful(
                sequence,
                navigation_state,
            )
        recurrent = hidden.reshape(batch * horizon, -1)
        distribution, values = self.decoder(recurrent)
        if self.recurrent_visual_safety is None:
            clearance_m, collision_risk = self._safety_estimates(
                flat_observations,
                recurrent,
            )
        else:
            clearance_m, collision_risk, next_safety_state = (
                self.recurrent_visual_safety.forward_train(
                    flat_observations,
                    batch=batch,
                    horizon=horizon,
                    state=safety_state,
                    terminals=terminals,
                )
            )
            if next_state is not None and next_safety_state is not None:
                next_state += next_safety_state
        return (
            self._gate_actions(
                distribution,
                flat_observations,
                None if clearance_m is None else clearance_m.detach(),
                None if collision_risk is None else collision_risk.detach(),
            ),
            values.reshape(batch, horizon),
            next_state,
            clearance_m,
            collision_risk,
        )

    def _safety_estimates(
        self,
        observations: torch.Tensor,
        recurrent_features: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.recurrent_safety is not None:
            return self.recurrent_safety(recurrent_features)
        if self.visual_safety is None:
            if self.clearance_head is None or self.collision_risk_head is None:
                return None, None
            features = self.encoder.vision_features(observations)
            return (
                4.0 * torch.sigmoid(self.clearance_head(features)),
                torch.sigmoid(self.collision_risk_head(features)),
            )
        return self.visual_safety(observations)

    def freeze_visual_safety_encoder(self) -> None:
        if self.recurrent_visual_safety is not None:
            for parameter in self.recurrent_visual_safety.parameters():
                parameter.requires_grad_(False)
        if self.recurrent_safety is not None:
            for parameter in self.recurrent_safety.parameters():
                parameter.requires_grad_(False)
        if self.visual_safety is not None:
            for parameter in self.visual_safety.parameters():
                parameter.requires_grad_(False)
        for head in (self.clearance_head, self.collision_risk_head):
            if head is not None:
                for parameter in head.parameters():
                    parameter.requires_grad_(False)

    def _split_state(self, state):
        if state is None:
            return None, None
        if self.recurrent_visual_safety is None:
            return state, None
        return state[:1], state[1:]

    def _gate_actions(
        self,
        distribution: torch.distributions.Normal,
        observations: torch.Tensor,
        clearance_m: torch.Tensor | None,
        collision_risk: torch.Tensor | None,
    ) -> torch.distributions.Normal:
        acquired, memory_bearing = self.encoder.target_memory_direction(observations)
        values = observations.reshape(observations.shape[0], -1).float()
        state = values[:, self.encoder.layout.proprioception_slice]
        confidence = state[
            :, GROUNDING_CONFIDENCE_INDEX : GROUNDING_CONFIDENCE_INDEX + 1
        ]
        horizontal_error = state[
            :, GROUNDING_HORIZONTAL_ERROR_INDEX : GROUNDING_HORIZONTAL_ERROR_INDEX + 1
        ]
        return project_semantic_actions(
            distribution,
            action_mode=self.action_mode,
            acquired=acquired,
            memory_bearing=memory_bearing,
            confidence=confidence,
            horizontal_error=horizontal_error,
            clearance_m=clearance_m,
            collision_risk=collision_risk,
        )
