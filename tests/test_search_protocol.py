import numpy as np
from flightrl.fleet.cooperative.search import SearchProtocol, visible
from flightrl.fleet.cooperative.mission import run


def test_confirmation_requires_delivered_scout_report():
    protocol = SearchProtocol(1)
    assert not protocol.eligible(2, 0, 0)
    blocked = np.array([[0.1, 0.4, -1, 1, 0, 2]])
    assert not visible(np.array([1.0, 0, 1]), np.array([0.0, 0, 0.65]), blocked)
    event = protocol.inspect(0, 0, 1.0, np.array([0.0, 0, 1.0]), np.array([0.0, 0]), [])
    assert event["type"] == "detected"
    assert not protocol.eligible(0, 0, 1.1)
    assert not protocol.eligible(2, 0, 1.19)
    assert protocol.eligible(2, 0, 1.2)
    roof = np.array([[-1, 1, -1, 1, 1.2, 1.4]])
    assert protocol.inspect(2, 0, 2, np.array([0.0, 0, 2.6]), np.zeros(2), roof) is None
    assert protocol.inspect(2, 0, 2, np.array([1.0, 0, 1.0]), np.zeros(2), []) is None
    assert (
        protocol.inspect(2, 0, 2, np.array([0.0, 0, 1.0]), np.zeros(2), [])["type"]
        == "confirmed"
    )


def test_search_mission_has_role_gates_and_vertical_motion():
    replay = run(140, mission="search_rescue", failure_s=None)
    assert replay["result"]["status"] == "complete"
    detections = {e["task"]: e for e in replay["events"] if e["type"] == "detected"}
    confirmations = [e for e in replay["events"] if e["type"] == "confirmed"]
    assert len(detections) == len(confirmations) == 9
    for event in confirmations:
        assert event["drone"] == 2
        assert detections[event["task"]]["drone"] < 2
        assert event["time_s"] >= detections[event["task"]]["time_s"] + 0.2
    positions = np.array([f["positions"] for f in replay["records"]])
    assert np.all(np.ptp(positions[:, :, 2], axis=0) > 0.15)
