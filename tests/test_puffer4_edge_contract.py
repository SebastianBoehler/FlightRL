from __future__ import annotations

import pytest

from flightrl.puffer4_edge_contract import (
    EDGE_ACTION_DIM,
    EDGE_FRAME_PIXELS,
    EDGE_MISSION_TOKEN_COUNT,
    EDGE_OBSERVATION_DIM,
    EDGE_TARGET_VOCABULARY,
    EDGE_TELEMETRY_DIM,
    edge_policy_contract_report,
    edge_target_id,
    edge_target_id_for_scene_object,
    edge_target_one_hot,
    validate_normalized_edge_action,
    validate_normalized_edge_telemetry,
    verify_edge_policy_contract,
)
from flightrl.puffer4_edge_wire import EdgeTimingProfile


def test_edge_contract_is_single_frame_and_host_detector_free() -> None:
    report = edge_policy_contract_report(hidden_size=48)

    verify_edge_policy_contract(report, hidden_size=48)

    observation = report["observation"]
    assert report["contract_id"] == "aideck-navigation-policy-v3"
    assert observation["flat_values"] == EDGE_OBSERVATION_DIM
    assert observation["frame"]["pixels"] == EDGE_FRAME_PIXELS
    assert observation["frame"]["packed_gray4_bytes"] == 1536
    assert observation["frame"]["history_planes"] == 1
    assert "phase" not in observation["segments"]
    assert "detector_evidence" not in observation["segments"]
    assert observation["telemetry"]["values"] == EDGE_TELEMETRY_DIM
    assert observation["mission_token"]["count"] == EDGE_MISSION_TOKEN_COUNT
    assert observation["mission_token"]["vocabulary"] == {
        "0": "door",
        "1": "monitor",
        "2": "sink",
    }
    assert report["action"]["order"] == ["vx", "vy", "vz", "yaw_rate"]
    assert report["runtime"]["training"] == "mac_edge_shaped_pytorch_reference"
    assert report["runtime"]["exact_deployment_graph_available"] is False
    assert report["runtime"]["exact_training_authority"] is False
    assert report["runtime"]["timing"]["status"] == "unmeasured_blocker"
    absent_target = report["model"]["grounding_label_semantics"]["absent_target"]
    assert "visible label zero and supervised" in absent_target
    assert "losses are masked" in absent_target


def test_edge_contract_fully_specifies_normalization_and_wire_layout() -> None:
    report = edge_policy_contract_report(hidden_size=48)

    telemetry = report["observation"]["telemetry"]
    assert telemetry["order"] == [field["name"] for field in telemetry["fields"]]
    assert all(
        {"name", "source_unit", "scale", "clip", "reference_frame", "formula"}
        <= field.keys()
        for field in telemetry["fields"]
    )
    assert "last_stm32_applied_vx_m_s" in telemetry["fields"][-4]["formula"]
    assert report["observation"]["previous_action"]["initial_value"] == [0.0] * 4
    input_wire = report["wire"]["policy_input"]
    assert input_wire["endianness"] == "little"
    assert input_wire["packing"] == "packed_no_padding"
    assert [field["name"] for field in input_wire["fields"]] == [
        "protocol_version",
        "flags",
        "frame_sequence",
        "capture_time_us",
        "telemetry_time_us",
        "mission_epoch",
        "arming_epoch",
        "target_id",
        "telemetry",
        "current_gray4",
    ]
    assert input_wire["bytes"] == 1_635
    assert input_wire["fields"][7]["allowed_ids"] == [0, 1, 2]
    assert report["wire"]["policy_output"]["fields"][6]["allowed_ids"] == [0, 1, 2]
    for wire_record in report["wire"].values():
        offset = 0
        for field in wire_record["fields"]:
            assert field["offset"] == offset
            offset += field["bytes"]
        assert offset == wire_record["bytes"]
    assert report["action"]["physical_mapping"][0]["scale"] == 0.25
    assert report["action"]["physical_mapping"][2]["scale"] == 0.15
    assert report["mission_boundary"]["goal_reached_owner"] == "stm32_mission_supervisor"


def test_edge_contract_and_target_encoding_are_deterministic() -> None:
    first = edge_policy_contract_report(hidden_size=48)
    second = edge_policy_contract_report(hidden_size=48)

    assert first == second
    assert EDGE_TARGET_VOCABULARY == ("door", "monitor", "sink")
    assert [edge_target_id(target) for target in EDGE_TARGET_VOCABULARY] == [0, 1, 2]
    assert edge_target_one_hot(1) == (0.0, 1.0, 0.0)
    try:
        edge_target_id("window")
    except ValueError as exc:
        assert "approved v3 target" in str(exc)
    else:
        raise AssertionError("unapproved target must be rejected")

    with pytest.raises(ValueError, match="integer"):
        edge_policy_contract_report(hidden_size=48.5)
    assert edge_target_id_for_scene_object(
        "door_7",
        {"door_7": "door"},
    ) == 0
    with pytest.raises(ValueError, match="explicit edge target binding"):
        edge_target_id_for_scene_object("door_7", {})


def test_edge_contract_accepts_only_measured_bound_runtime_timing() -> None:
    timing = EdgeTimingProfile(
        nominal_period_us=50_000,
        minimum_period_us=45_000,
        maximum_period_us=60_000,
        max_frame_telemetry_skew_us=5_000,
        max_proposal_age_us=80_000,
        measurement_frames=1_000,
        measurement_sha256="a" * 64,
    )

    report = edge_policy_contract_report(hidden_size=48, timing=timing)

    verify_edge_policy_contract(report, hidden_size=48, timing=timing)
    assert report["runtime"]["timing_bound"] is True
    assert report["runtime"]["exact_training_authority"] is False
    assert report["runtime"]["timing"]["status"] == "measured_and_bound"

    with pytest.raises(ValueError, match="at least 1000"):
        EdgeTimingProfile(
            nominal_period_us=50_000,
            minimum_period_us=45_000,
            maximum_period_us=60_000,
            max_frame_telemetry_skew_us=5_000,
            max_proposal_age_us=80_000,
            measurement_frames=999,
            measurement_sha256="a" * 64,
        )


def test_edge_wire_value_validators_reject_nonfinite_and_noncanonical_values() -> None:
    telemetry = [0.0] * EDGE_TELEMETRY_DIM
    telemetry[8] = 1.0
    telemetry[14] = 1.0
    assert validate_normalized_edge_telemetry(telemetry) == tuple(telemetry)
    assert validate_normalized_edge_action([0.0] * EDGE_ACTION_DIM) == (0.0,) * EDGE_ACTION_DIM
    with pytest.raises(ValueError, match="target ID"):
        edge_target_one_hot(True)
    with pytest.raises(ValueError, match="nonfinite"):
        validate_normalized_edge_telemetry([float("nan"), *([0.0] * (EDGE_TELEMETRY_DIM - 1))])
    with pytest.raises(ValueError, match="out of range"):
        validate_normalized_edge_action([0.0, 0.0, 0.0, 1.1])
    with pytest.raises(ValueError, match="nonfinite"):
        validate_normalized_edge_action(["0", 0.0, 0.0, 0.0])
