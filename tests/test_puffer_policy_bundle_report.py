from __future__ import annotations

from pathlib import Path

import torch

from flightrl.sixdof.mode_conditioned import expand_policy_for_modes
from flightrl.sixdof.puffer_policy import PufferPolicyMetadata, PufferSixDofPolicy
from scripts.build_puffer_policy_bundle_transfer_report import load_policy, render_markdown


def test_bundle_loader_accepts_plain_28d_policy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "plain.bin"
    torch.save(PufferSixDofPolicy(PufferPolicyMetadata(28, 16, 4, 1)).state_dict(), checkpoint)

    policy = load_policy(str(checkpoint), "obstacle_hover")

    assert policy.metadata.observation_dim == 28


def test_bundle_loader_wraps_mode_conditioned_30d_policy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "mode.bin"
    base = PufferSixDofPolicy(PufferPolicyMetadata(28, 16, 4, 1))
    torch.save(expand_policy_for_modes(base).state_dict(), checkpoint)

    policy = load_policy(str(checkpoint), "velocity_target")

    assert policy.metadata.observation_dim == 28
    assert policy(torch.zeros(2, 28)).shape == (2, 4)


def test_bundle_markdown_surfaces_obstacle_source_failures() -> None:
    markdown = render_markdown(
        {
            "passed": True,
            "safety": "offline only",
            "bundle": {
                "label": "bundle",
                "passed": True,
                "obstacle_checkpoint": "obstacle.bin",
                "velocity_checkpoint": "velocity.bin",
                "obstacle": {
                    "passed": True,
                    "sim": {},
                    "live_logs": {
                        "raw_run90": {
                            "failed_source": True,
                            "source_failure_evidence": {
                                "failures": ["source_precontact_drift"],
                                "source": {"precontact_horizontal_speed_max_m_s": 5.1},
                            },
                            "shadow": {
                                "scored_samples": 122,
                                "excluded_source_samples": 823,
                                "gate": {"failures": []},
                                "groups": {"all": {"l2_p95": 0.03}},
                            },
                            "command_gate": {"failures": [], "safe": {"action_abs_p95": 0.28}},
                            "crash_replay": {
                                "gate": {"failures": []},
                                "groups": {
                                    "all": {"l2_p95": 0.34},
                                    "precontact_drift": {"l2_p95": 0.22},
                                },
                            },
                        }
                    },
                },
                "velocity": {
                    "vel70": {
                        "gate": {"passed": True, "failures": []},
                        "policy": {"horizontal_l2_p95_m_s": 0.03, "yaw_abs_p95_deg_s": 4.0, "sign_agreement": {}},
                    }
                },
            },
        }
    )

    assert "## Obstacle Metrics" in markdown
    assert "source_precontact_drift" in markdown
    assert "5.1000" in markdown
    assert "122/823" in markdown
