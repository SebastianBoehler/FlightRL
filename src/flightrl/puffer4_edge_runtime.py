from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch

from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
)
from flightrl.puffer4_edge_contract import validate_normalized_edge_action
from flightrl.puffer4_edge_policy import EdgeNavigationActor
from flightrl.puffer4_edge_wire import (
    EdgeInputStamp,
    EdgeProposalStamp,
    EdgeTimingProfile,
    validate_edge_input_order,
)
from flightrl.puffer4_edge_wire_codec import (
    DecodedEdgeInput,
    decode_edge_input,
    encode_edge_proposal,
)


EDGE_POLICY_NOMINAL_PERIOD_US = round(
    1_000_000 * FIXED_DOOR_EVIDENCE_AGE_CONTRACT.control_dt_s
)


@dataclass(frozen=True, slots=True)
class EdgeRuntimeResult:
    proposal: bytes
    action: tuple[float, ...]
    grounding: tuple[float, ...]
    state_reset_applied: bool
    hidden_sha256: str


class EdgePolicyRuntime:
    """Fail-closed, non-actuating recurrent edge-v3 packet runtime."""

    def __init__(
        self,
        actor: EdgeNavigationActor,
        *,
        timing: EdgeTimingProfile,
        trained_target_ids: tuple[int, ...],
    ) -> None:
        if timing.nominal_period_us != EDGE_POLICY_NOMINAL_PERIOD_US:
            raise ValueError(
                "runtime timing must match the simulator training cadence "
                f"of {EDGE_POLICY_NOMINAL_PERIOD_US} us"
            )
        if not trained_target_ids or any(
            type(value) is not int or not 0 <= value < 3
            for value in trained_target_ids
        ):
            raise ValueError("runtime trained target IDs are invalid")
        if len(set(trained_target_ids)) != len(trained_target_ids):
            raise ValueError("runtime trained target IDs must be unique")
        self.actor = actor.eval()
        self.timing = timing
        self.trained_target_ids = frozenset(trained_target_ids)
        self._device = next(actor.parameters()).device
        self._state = actor.initial_state(1, device=self._device)
        self._previous_input: EdgeInputStamp | None = None
        self._previous_proposal_sequence: int | None = None
        self._requires_reset = True

    @property
    def requires_reset(self) -> bool:
        return self._requires_reset

    @property
    def hidden_state(self) -> torch.Tensor:
        return self._state.detach().clone()

    def process(self, packet: bytes | bytearray | memoryview) -> EdgeRuntimeResult:
        try:
            decoded = decode_edge_input(packet)
            result, next_state, proposal_sequence = self._process_validated(decoded)
        except Exception:
            self._requires_reset = True
            raise
        self._state = next_state.detach()
        self._previous_input = decoded.stamp
        self._previous_proposal_sequence = proposal_sequence
        self._requires_reset = False
        return result

    @torch.no_grad()
    def _process_validated(
        self,
        decoded: DecodedEdgeInput,
    ) -> tuple[EdgeRuntimeResult, torch.Tensor, int]:
        if decoded.stamp.target_id not in self.trained_target_ids:
            raise ValueError("edge target was not trained and evaluated")
        contract_reset = validate_edge_input_order(
            decoded.stamp,
            self._previous_input,
            self.timing,
        )
        reset_applied = bool(
            decoded.reset_requested or contract_reset or self._requires_reset
        )
        if (contract_reset or self._requires_reset) and not decoded.reset_requested:
            raise ValueError("edge input must explicitly request recurrent reset")
        state = (
            self.actor.initial_state(1, device=self._device)
            if reset_applied
            else self._state.detach().clone()
        )
        observation = decoded.model_observation().to(self._device)
        action_tensor, grounding_tensor, next_state = self.actor.forward_step(
            observation,
            state,
        )
        _validate_actor_outputs(
            action_tensor,
            grounding_tensor,
            next_state,
            hidden_size=self.actor.hidden_size,
            device=self._device,
        )
        action = validate_normalized_edge_action(
            action_tensor[0].detach().cpu().tolist()
        )
        grounding = _validated_grounding(grounding_tensor[0])
        proposal_sequence = (
            0
            if self._previous_proposal_sequence is None
            else (self._previous_proposal_sequence + 1) & 0xFFFF_FFFF
        )
        stamp = EdgeProposalStamp(
            source_frame_sequence=decoded.stamp.frame_sequence,
            source_capture_time_us=decoded.stamp.capture_time_us,
            mission_epoch=decoded.stamp.mission_epoch,
            arming_epoch=decoded.stamp.arming_epoch,
            source_target_id=decoded.stamp.target_id,
            proposal_sequence=proposal_sequence,
        )
        proposal = encode_edge_proposal(
            stamp,
            state_reset_applied=reset_applied,
            action=action,
        )
        return (
            EdgeRuntimeResult(
                proposal=proposal,
                action=action,
                grounding=grounding,
                state_reset_applied=reset_applied,
                hidden_sha256=_tensor_sha256(next_state),
            ),
            next_state,
            proposal_sequence,
        )


def _validated_grounding(value: torch.Tensor) -> tuple[float, ...]:
    if value.shape != (4,) or value.dtype != torch.float32:
        raise RuntimeError("edge grounding violates the runtime shape or dtype")
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError("edge grounding is nonfinite")
    result = tuple(float(item) for item in value.detach().cpu())
    if not (
        0.0 <= result[0] <= 1.0
        and all(-1.0 <= item <= 1.0 for item in result[1:3])
        and 0.0 <= result[3] <= 1.0
    ):
        raise RuntimeError("edge grounding is outside contract bounds")
    return result


def _validate_actor_outputs(
    action: torch.Tensor,
    grounding: torch.Tensor,
    state: torch.Tensor,
    *,
    hidden_size: int,
    device: torch.device,
) -> None:
    expected = ((1, 4), (1, 4), (1, hidden_size))
    for label, value, shape in zip(
        ("action", "grounding", "hidden state"),
        (action, grounding, state),
        expected,
        strict=True,
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != shape
            or value.dtype != torch.float32
            or value.device != device
        ):
            raise RuntimeError(f"edge actor {label} violates shape, dtype, or device")
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"edge actor {label} is nonfinite")
    if bool(torch.any((state < 0.0) | (state > 6.0))):
        raise RuntimeError("edge actor hidden state violates the [0, 6] invariant")


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()
