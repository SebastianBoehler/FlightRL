from __future__ import annotations

from types import SimpleNamespace

import pytest

from flightrl.sixdof.transfer_test import LiveLogCase, TransferTestConfig
from scripts import build_puffer_policy_bundle_transfer_report as bundle_report


def test_transfer_config_defaults_to_live_previous_action_scale() -> None:
    assert TransferTestConfig().previous_action_observation_scale == pytest.approx(0.25)


def test_bundle_report_passes_previous_action_scale_to_sim_and_crash(monkeypatch) -> None:
    seen: dict[str, float] = {}

    def fake_eval(_policy, config):
        seen["sim_scale"] = config.previous_action_observation_scale
        return {"python": {"gate": {"passed": True, "failures": []}}}

    def fake_rows(_path):
        return [{"stateEstimate.z": 0.5, "range.zrange": 500.0}]

    def fake_shadow(_policy, _rows, _case, _config):
        return {"gate": {"passed": True, "failures": []}}

    def fake_raw_rows(_policy, _rows, _config):
        return [
            {
                "stateEstimate.z": 0.5,
                "range.zrange": 500.0,
                "range.front": 800.0,
                "range.back": 900.0,
                "range.left": 700.0,
                "range.right": 600.0,
                "sys.canfly": 1.0,
                "sys.isTumbled": 0.0,
                "thrust_percent": 50.0,
                "roll_rate_deg_s": 0.0,
                "pitch_rate_deg_s": 10.0,
                "commander_pitch_rate_deg_s": -10.0,
                "yaw_rate_deg_s": 0.0,
                "action_thrust": 0.0,
                "action_roll_rate": 0.0,
                "action_pitch_rate": 0.0,
                "action_yaw_rate": 0.0,
            }
        ]

    def fake_crash(_policy, _rows, _config, *, previous_action_observation_scale):
        seen["crash_scale"] = previous_action_observation_scale
        return {"gate": {"passed": True, "failures": []}}

    monkeypatch.setattr(bundle_report, "evaluate_puffer_backends", fake_eval)
    monkeypatch.setattr(bundle_report, "load_live_rows", fake_rows)
    monkeypatch.setattr(bundle_report, "live_shadow_report", fake_shadow)
    monkeypatch.setattr(bundle_report, "raw_shadow_rows", fake_raw_rows)
    monkeypatch.setattr(bundle_report, "score_crash_replay_policy", fake_crash)

    config = TransferTestConfig(min_command_safe_rows=1)
    report = bundle_report.obstacle_report(SimpleNamespace(), "checkpoint.bin", [LiveLogCase("failed", "log.csv", True)], config)

    assert report["passed"] is True
    assert seen == {"sim_scale": pytest.approx(0.25), "crash_scale": pytest.approx(0.25)}
