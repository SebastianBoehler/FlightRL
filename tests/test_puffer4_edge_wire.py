from __future__ import annotations

import pytest

from flightrl.puffer4_edge_wire import (
    EdgeInputStamp,
    EdgeProposalStamp,
    EdgeTimingProfile,
    forward_u32_delta,
    validate_edge_input_order,
    validate_edge_proposal_stamp,
)


@pytest.fixture
def timing() -> EdgeTimingProfile:
    return EdgeTimingProfile(
        nominal_period_us=50_000,
        minimum_period_us=40_000,
        maximum_period_us=60_000,
        max_frame_telemetry_skew_us=4_000,
        max_proposal_age_us=80_000,
        measurement_frames=2_000,
        measurement_sha256="b" * 64,
    )


def _input(**changes: int) -> EdgeInputStamp:
    values = {
        "frame_sequence": 10,
        "capture_time_us": 1_000_000,
        "telemetry_time_us": 998_000,
        "mission_epoch": 3,
        "arming_epoch": 8,
        "target_id": 0,
    }
    return EdgeInputStamp(**(values | changes))


def _proposal(**changes: int) -> EdgeProposalStamp:
    values = {
        "source_frame_sequence": 10,
        "source_capture_time_us": 1_000_000,
        "mission_epoch": 3,
        "arming_epoch": 8,
        "source_target_id": 0,
        "proposal_sequence": 7,
    }
    return EdgeProposalStamp(**(values | changes))


def test_unsigned_ordering_accepts_wrap_and_rejects_old_values() -> None:
    assert forward_u32_delta(2, 0xFFFF_FFFE, "sequence") == 4
    with pytest.raises(ValueError, match="duplicate or reordered"):
        forward_u32_delta(9, 10, "sequence")
    with pytest.raises(ValueError, match="duplicate or reordered"):
        forward_u32_delta(10, 10, "sequence")


def test_input_order_requires_alignment_and_resets_on_drop(
    timing: EdgeTimingProfile,
) -> None:
    assert validate_edge_input_order(_input(), None, timing) is True
    assert validate_edge_input_order(
        _input(
            frame_sequence=11,
            capture_time_us=1_050_000,
            telemetry_time_us=1_049_000,
        ),
        _input(),
        timing,
    ) is False
    assert validate_edge_input_order(
        _input(
            frame_sequence=12,
            capture_time_us=1_050_000,
            telemetry_time_us=1_049_000,
        ),
        _input(),
        timing,
    ) is True
    with pytest.raises(ValueError, match="misaligned"):
        validate_edge_input_order(
            _input(telemetry_time_us=990_000),
            None,
            timing,
        )


@pytest.mark.parametrize(
    ("capture_delta_us", "reset_required"),
    ((39_999, True), (40_000, False), (60_000, False), (60_001, True)),
)
def test_input_order_enforces_the_full_measured_capture_period(
    timing: EdgeTimingProfile,
    capture_delta_us: int,
    reset_required: bool,
) -> None:
    current_capture = 1_000_000 + capture_delta_us

    assert validate_edge_input_order(
        _input(
            frame_sequence=11,
            capture_time_us=current_capture,
            telemetry_time_us=current_capture - 1_000,
        ),
        _input(),
        timing,
    ) is reset_required


def test_stm32_accepts_only_fresh_exactly_echoed_new_proposals(
    timing: EdgeTimingProfile,
) -> None:
    assert validate_edge_proposal_stamp(
        _proposal(),
        _input(),
        previous_proposal_sequence=6,
        stm32_now_us=1_050_000,
        timing=timing,
        state_reset_applied=False,
        input_reset_required=False,
    ) == 50_000
    with pytest.raises(ValueError, match="latest input authority"):
        validate_edge_proposal_stamp(
            _proposal(source_target_id=1),
            _input(),
            previous_proposal_sequence=6,
            stm32_now_us=1_050_000,
            timing=timing,
            state_reset_applied=False,
            input_reset_required=False,
        )
    with pytest.raises(ValueError, match="duplicate or reordered"):
        validate_edge_proposal_stamp(
            _proposal(proposal_sequence=6),
            _input(),
            previous_proposal_sequence=6,
            stm32_now_us=1_050_000,
            timing=timing,
            state_reset_applied=False,
            input_reset_required=False,
        )
    with pytest.raises(ValueError, match="stale or from the future"):
        validate_edge_proposal_stamp(
            _proposal(),
            _input(),
            previous_proposal_sequence=None,
            stm32_now_us=1_100_000,
            timing=timing,
            state_reset_applied=False,
            input_reset_required=False,
        )


@pytest.mark.parametrize(
    ("state_reset_applied", "input_reset_required"),
    ((True, False), (False, True), (1, True), (True, 1)),
)
def test_stm32_rejects_wrong_or_nonboolean_reset_acknowledgement(
    timing: EdgeTimingProfile,
    state_reset_applied: object,
    input_reset_required: object,
) -> None:
    with pytest.raises(ValueError, match="reset acknowledgement"):
        validate_edge_proposal_stamp(
            _proposal(),
            _input(),
            previous_proposal_sequence=6,
            stm32_now_us=1_050_000,
            timing=timing,
            state_reset_applied=state_reset_applied,
            input_reset_required=input_reset_required,
        )


@pytest.mark.parametrize("reset_required", (False, True))
def test_stm32_accepts_exact_reset_acknowledgement(
    timing: EdgeTimingProfile,
    reset_required: bool,
) -> None:
    assert validate_edge_proposal_stamp(
        _proposal(),
        _input(),
        previous_proposal_sequence=6,
        stm32_now_us=1_050_000,
        timing=timing,
        state_reset_applied=reset_required,
        input_reset_required=reset_required,
    ) == 50_000


@pytest.mark.parametrize("target_id", (-1, 3, 255, True))
def test_wire_stamps_reject_targets_outside_v3_vocabulary(
    target_id: object,
) -> None:
    with pytest.raises(ValueError, match="approved v3 vocabulary"):
        _input(target_id=target_id)
    with pytest.raises(ValueError, match="approved v3 vocabulary"):
        _proposal(source_target_id=target_id)
