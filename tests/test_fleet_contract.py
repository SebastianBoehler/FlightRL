import numpy as np
from flightrl.fleet.communication import PeerLink
from flightrl.fleet.vehicles import VEHICLES


def test_peer_delay_expiry_and_no_self_messages():
    link = PeerLink(3)
    positions = np.arange(9).reshape(3, 3)
    link.publish(0, positions, np.zeros((3, 3)), [0, 1, 2], [set(), set(), set()])
    assert link.receive(0.1) == [[], [], []]
    received = link.receive(0.2)
    assert all(len(row) == 2 for row in received)
    assert all(m.sender != i for i, row in enumerate(received) for m in row)
    assert link.receive(1.1) == [[], [], []]


def test_vehicle_envelopes_and_mass_are_not_visual_scaling():
    assert (
        VEHICLES["fpv"].radius
        < VEHICLES["industrial"].radius
        < VEHICLES["agriculture"].radius
    )
    for v in VEHICLES.values():
        assert v.physics()[0] == np.float32(v.mass_kg)
        assert v.physics()[8] == np.float32(v.motor_tau_s)
