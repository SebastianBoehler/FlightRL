from __future__ import annotations

import pytest
import torch

from flightrl.puffer4_edge_policy import EdgeNavigationActor
from flightrl.puffer4_edge_runtime import (
    EDGE_POLICY_NOMINAL_PERIOD_US,
    EdgePolicyRuntime,
)
from flightrl.puffer4_edge_wire import EdgeInputStamp, EdgeTimingProfile
from flightrl.puffer4_edge_wire_codec import (
    decode_edge_proposal,
    encode_edge_input,
)


def _timing() -> EdgeTimingProfile:
    return EdgeTimingProfile(
        nominal_period_us=EDGE_POLICY_NOMINAL_PERIOD_US,
        minimum_period_us=14_000,
        maximum_period_us=20_000,
        max_frame_telemetry_skew_us=5_000,
        max_proposal_age_us=30_000,
        measurement_frames=1_000,
        measurement_sha256="a" * 64,
    )


def _runtime() -> EdgePolicyRuntime:
    torch.manual_seed(17)
    return EdgePolicyRuntime(
        EdgeNavigationActor(hidden_size=48),
        timing=_timing(),
        trained_target_ids=(0,),
    )


def _packet(
    sequence: int,
    *,
    reset: bool,
    target_id: int = 0,
    mission_epoch: int = 1,
    capture_time_us: int | None = None,
) -> bytes:
    telemetry = [0.0] * 19
    telemetry[8] = 1.0
    telemetry[14] = 1.0
    capture = (
        capture_time_us
        if capture_time_us is not None
        else 100_000 + sequence * EDGE_POLICY_NOMINAL_PERIOD_US
    )
    return encode_edge_input(
        EdgeInputStamp(
            frame_sequence=sequence,
            capture_time_us=capture,
            telemetry_time_us=capture,
            mission_epoch=mission_epoch,
            arming_epoch=1,
            target_id=target_id,
        ),
        reset_requested=reset,
        telemetry=telemetry,
        packed_gray4=bytes(1536),
    )


def test_runtime_requires_explicit_boot_reset_then_commits_atomically() -> None:
    runtime = _runtime()
    initial = runtime.hidden_state

    with pytest.raises(ValueError, match="explicitly request"):
        runtime.process(_packet(1, reset=False))

    assert torch.equal(runtime.hidden_state, initial)
    assert runtime.requires_reset is True

    first = runtime.process(_packet(1, reset=True))
    decoded = decode_edge_proposal(first.proposal)

    assert first.state_reset_applied is True
    assert decoded.state_reset_applied is True
    assert decoded.stamp.proposal_sequence == 0
    assert not torch.equal(runtime.hidden_state, initial)
    assert runtime.requires_reset is False


def test_runtime_rejects_timing_profile_from_nontraining_cadence() -> None:
    timing = EdgeTimingProfile(
        nominal_period_us=50_000,
        minimum_period_us=45_000,
        maximum_period_us=60_000,
        max_frame_telemetry_skew_us=5_000,
        max_proposal_age_us=80_000,
        measurement_frames=1_000,
        measurement_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="training cadence"):
        EdgePolicyRuntime(
            EdgeNavigationActor(hidden_size=48),
            timing=timing,
            trained_target_ids=(0,),
        )


def test_runtime_carries_state_and_increments_proposal_sequence() -> None:
    runtime = _runtime()
    runtime.process(_packet(1, reset=True))
    first_state = runtime.hidden_state

    second = runtime.process(_packet(2, reset=False))

    assert second.state_reset_applied is False
    assert decode_edge_proposal(second.proposal).stamp.proposal_sequence == 1
    assert not torch.equal(runtime.hidden_state, first_state)


def test_duplicate_rejection_does_not_commit_and_forces_next_reset() -> None:
    runtime = _runtime()
    runtime.process(_packet(1, reset=True))
    before = runtime.hidden_state

    with pytest.raises(ValueError, match="duplicate or reordered"):
        runtime.process(_packet(1, reset=False))

    assert torch.equal(runtime.hidden_state, before)
    assert runtime.requires_reset is True
    with pytest.raises(ValueError, match="explicitly request"):
        runtime.process(_packet(2, reset=False))
    accepted = runtime.process(_packet(2, reset=True))
    assert accepted.state_reset_applied is True


def test_epoch_or_dropped_frame_requires_explicit_reset() -> None:
    runtime = _runtime()
    runtime.process(_packet(1, reset=True))

    with pytest.raises(ValueError, match="explicitly request"):
        runtime.process(_packet(2, reset=False, mission_epoch=2))
    runtime.process(_packet(2, reset=True, mission_epoch=2))

    with pytest.raises(ValueError, match="explicitly request"):
        runtime.process(
            _packet(4, reset=False, mission_epoch=2, capture_time_us=350_000)
        )


def test_untrained_target_and_inference_error_never_commit_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    runtime.process(_packet(1, reset=True))
    before = runtime.hidden_state

    with pytest.raises(ValueError, match="not trained"):
        runtime.process(_packet(2, reset=True, target_id=1))
    assert torch.equal(runtime.hidden_state, before)

    def fail(*_args, **_kwargs):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(runtime.actor, "forward_step", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        runtime.process(_packet(2, reset=True))
    assert torch.equal(runtime.hidden_state, before)
    assert runtime.requires_reset is True


@pytest.mark.parametrize(
    "invalid_output",
    ("action_batch", "grounding_dtype", "state_shape", "state_nonfinite", "state_range"),
)
def test_invalid_actor_output_never_commits_or_advances_sequence(
    monkeypatch: pytest.MonkeyPatch,
    invalid_output: str,
) -> None:
    runtime = _runtime()
    runtime.process(_packet(1, reset=True))
    before = runtime.hidden_state
    original = runtime.actor.forward_step

    def invalid(observation, state):
        action = torch.zeros(1, 4)
        grounding = torch.zeros(1, 4)
        next_state = torch.zeros(1, 48)
        if invalid_output == "action_batch":
            action = torch.zeros(2, 4)
        elif invalid_output == "grounding_dtype":
            grounding = grounding.double()
        elif invalid_output == "state_shape":
            next_state = torch.zeros(2, 48)
        elif invalid_output == "state_nonfinite":
            next_state[0, 0] = float("nan")
        else:
            next_state[0, 0] = 6.01
        return action, grounding, next_state

    monkeypatch.setattr(runtime.actor, "forward_step", invalid)
    with pytest.raises(RuntimeError, match="edge actor"):
        runtime.process(_packet(2, reset=True))
    assert torch.equal(runtime.hidden_state, before)
    assert runtime.requires_reset is True

    monkeypatch.setattr(runtime.actor, "forward_step", original)
    accepted = runtime.process(_packet(2, reset=True))
    assert decode_edge_proposal(accepted.proposal).stamp.proposal_sequence == 1
