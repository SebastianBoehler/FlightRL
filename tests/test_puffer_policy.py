from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import torch

from flightrl.sixdof.checkpoint_contract import PUFFER_POLICY_FORMAT, build_checkpoint_payload
from flightrl.sixdof.puffer_policy import PufferPolicyMetadata, PufferSixDofPolicy, infer_metadata, load_puffer_sixdof_policy


ROOT = Path(__file__).resolve().parents[1]


def test_puffer_policy_infers_metadata_and_loads_state_dict() -> None:
    original = PufferSixDofPolicy(PufferPolicyMetadata(observation_dim=28, hidden_size=8, action_dim=4, num_layers=2))
    loaded = PufferSixDofPolicy.from_state_dict(original.state_dict())

    assert infer_metadata(original.state_dict()).observation_dim == 28
    assert loaded.metadata.hidden_size == 8
    assert loaded.metadata.action_dim == 4


def test_puffer_policy_tanh_maps_raw_action_location() -> None:
    policy = PufferSixDofPolicy(PufferPolicyMetadata(observation_dim=3, hidden_size=4, action_dim=2, num_layers=1))
    with torch.no_grad():
        policy.decoder.decoder_mean.bias[:] = torch.tensor([3.0, -3.0])

    actions = policy(torch.zeros(5, 3))

    assert actions.shape == (5, 2)
    expected = torch.tanh(policy.action_location(torch.zeros(5, 3)))
    torch.testing.assert_close(actions, expected)
    assert torch.max(actions) < 1.0
    assert torch.min(actions) > -1.0


def test_puffer_checkpoint_loader_rejects_raw_uncontracted_state_dict(
    tmp_path: Path,
) -> None:
    path = tmp_path / "raw.bin"
    policy = PufferSixDofPolicy(
        PufferPolicyMetadata(
            observation_dim=28,
            hidden_size=8,
            action_dim=4,
            num_layers=2,
        )
    )
    torch.save(policy.state_dict(), path)

    with pytest.raises(ValueError, match="legacy checkpoints are rejected"):
        load_puffer_sixdof_policy(str(path))


def test_puffer_checkpoint_loader_accepts_contracted_wrapper(tmp_path: Path) -> None:
    path = tmp_path / "current.bin"
    policy = PufferSixDofPolicy(
        PufferPolicyMetadata(
            observation_dim=28,
            hidden_size=8,
            action_dim=4,
            num_layers=2,
        )
    )
    torch.save(
        build_checkpoint_payload(
            state_dict=policy.state_dict(),
            tasks=("position_yaw",),
            hidden_size=8,
            checkpoint_format=PUFFER_POLICY_FORMAT,
        ),
        path,
    )

    loaded = load_puffer_sixdof_policy(str(path))

    assert loaded.checkpoint_metadata is not None
    assert loaded.checkpoint_metadata.tasks == ("position_yaw",)


def test_puffer_wrapper_binds_raw_state_dict_to_declared_contract(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.bin"
    wrapped = tmp_path / "wrapped.pt"
    policy = PufferSixDofPolicy(
        PufferPolicyMetadata(
            observation_dim=28,
            hidden_size=8,
            action_dim=4,
            num_layers=2,
        )
    )
    torch.save(policy.state_dict(), raw)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "wrap_puffer_sixdof_checkpoint.py"),
            "--raw-checkpoint",
            str(raw),
            "--output",
            str(wrapped),
            "--task",
            "position_yaw",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    loaded = load_puffer_sixdof_policy(str(wrapped))
    payload = torch.load(wrapped, map_location="cpu")
    assert loaded.checkpoint_metadata is not None
    assert payload["source_raw_checkpoint"]["sha256"]
    assert payload["trainer"] == "pufferlib_external"
