from __future__ import annotations

import pytest

from flightrl.puffer4_door_stream_contract import (
    door_stream_contract_report,
    verify_door_stream_contract,
)


def test_stream_contract_binds_episode_seed_inputs_and_group_schema() -> None:
    report = door_stream_contract_report()

    verify_door_stream_contract(report)

    assert report["contract_id"] == "fixed-door-episode-stream-v1"
    assert report["physical_seed_inputs"] == [
        "base_seed",
        "environment_index",
        "episode_index_u64",
    ]
    assert report["appearance_seed_inputs"] == [
        "base_seed",
        "appearance_seed",
        "environment_index",
        "episode_index_u64",
    ]
    assert report["group_schema"]["kind"] == "marginal_not_joint"


def test_stream_contract_rejects_mutation() -> None:
    report = door_stream_contract_report()
    report["group_schema"]["version"] = 2

    with pytest.raises(ValueError, match="SHA-256"):
        verify_door_stream_contract(report)
