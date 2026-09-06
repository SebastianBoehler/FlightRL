"""Guard direct inference against export drift and classical-controller substitution."""

import numpy as np
import torch
from flightrl import _binding
from flightrl.fleet.flight_policy.model import Network, Policy, samples
from flightrl.fleet.flight_policy.course import run


def test_numpy_policy_matches_trained_network_format(tmp_path):
    torch.manual_seed(9)
    network = Network()
    checkpoint = tmp_path / "controller.pt"
    torch.save({"model": network.state_dict()}, checkpoint)
    policy = Policy(checkpoint)
    x, _ = samples(81, 16)
    out = x
    for i, (w, b) in enumerate(policy.layers):
        out = out @ w + b
        if i < len(policy.layers) - 1:
            out = np.tanh(out)
    with torch.no_grad():
        expected = network(torch.tensor(x)).numpy()
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_course_uses_supplied_thrust_without_classical_correction(monkeypatch):
    def forbidden(*args):
        raise AssertionError("Unexpected classical control")

    monkeypatch.setattr(_binding, "sixdof_setpoint_actions", forbidden)
    # Zero collective must fall and terminate, never be rescued by the teacher.
    result = run(lambda d, v, q, h: np.tile([-1.0, 0, 0, 0], (3, 1)), 180)
    assert result["result"]["status"] == "collision"
    assert result["records"][-1]["positions"][0][2] < 0.4
