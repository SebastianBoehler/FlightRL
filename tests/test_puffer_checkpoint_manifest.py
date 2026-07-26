from __future__ import annotations

import json
import subprocess
import sys

from flightrl.sim2real.hardware_approval import hardware_approval_status
from flightrl.sim2real.puffer_checkpoint_manifest import build_puffer_checkpoint_manifest


def test_puffer_manifest_blocks_deployment_ready_records_when_transfer_blocked(tmp_path) -> None:
    gate = write_json(
        tmp_path / "gate.json",
        {"transfer_approved": False, "summary": {"failures": ["range_deck_damaged"]}},
    )
    bundle = write_json(tmp_path / "bundle.json", bundle_report())

    report = build_puffer_checkpoint_manifest(transfer_gate=gate, bundle_report=bundle)

    assert report["transfer_approved"] is False
    assert report["transfer_failures"] == ["range_deck_damaged"]
    assert report["summary"] == {"total": 2, "sim_ready": 2, "deployment_ready": 2, "hardware_approved": 0}
    assert [item["status"] for item in report["records"]] == [
        "deployment_ready_but_transfer_blocked",
        "deployment_ready_but_transfer_blocked",
    ]
    assert report["inputs"]["bundle_report"]["sha256"]


def test_puffer_manifest_approves_bundle_records_only_when_transfer_passes(tmp_path) -> None:
    gate = write_json(tmp_path / "gate.json", {"transfer_approved": True, "summary": {"failures": []}})
    bundle = write_json(tmp_path / "bundle.json", bundle_report())

    report = build_puffer_checkpoint_manifest(transfer_gate=gate, bundle_report=bundle)

    assert report["summary"]["hardware_approved"] == 2
    assert all(record["status"] == "hardware_approved" for record in report["records"])


def test_puffer_manifest_keeps_failed_velocity_component_blocked(tmp_path) -> None:
    gate = write_json(tmp_path / "gate.json", {"transfer_approved": True})
    bundle = bundle_report()
    bundle["bundle"]["velocity"]["vel70"]["gate"] = {"passed": False, "failures": ["velocity_yaw_abs_p95"]}
    bundle_path = write_json(tmp_path / "bundle.json", bundle)

    report = build_puffer_checkpoint_manifest(transfer_gate=gate, bundle_report=bundle_path)

    velocity = next(record for record in report["records"] if record["task"] == "velocity_target")
    assert velocity["hardware_approved"] is False
    assert velocity["deployment_failures"] == ["puffer_velocity_transfer_failed"]
    assert report["summary"]["hardware_approved"] == 1


def test_puffer_manifest_cli_writes_json_and_markdown(tmp_path) -> None:
    gate = write_json(tmp_path / "gate.json", {"transfer_approved": False})
    bundle = write_json(tmp_path / "bundle.json", bundle_report())
    output = tmp_path / "manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_puffer_checkpoint_manifest.py",
            "--transfer-gate",
            str(gate),
            "--bundle-report",
            str(bundle),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "hardware_approved=0" in result.stdout
    assert output.exists()
    assert output.with_suffix(".md").exists()
    assert json.loads(output.read_text())["bundle_report"] == str(bundle)


def test_puffer_manifest_integrates_with_hardware_approval_status(tmp_path) -> None:
    gate = write_json(tmp_path / "gate.json", {"transfer_approved": False})
    bundle = write_json(tmp_path / "bundle.json", bundle_report(obstacle_checkpoint=str(tmp_path / "obstacle.bin")))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(build_puffer_checkpoint_manifest(transfer_gate=gate, bundle_report=bundle)))

    status = hardware_approval_status(tmp_path / "obstacle.bin", manifest)

    assert status["hardware_approved"] is False
    assert status["approval_status"] == "blocked"
    assert "transfer gate is not approved" in status["approval_error"]


def bundle_report(
    *,
    obstacle_checkpoint: str = "artifacts/checkpoints/obstacle.bin",
    velocity_checkpoint: str = "artifacts/checkpoints/velocity.bin",
):
    return {
        "bundle": {
            "label": "candidate_bundle",
            "obstacle_checkpoint": obstacle_checkpoint,
            "velocity_checkpoint": velocity_checkpoint,
            "obstacle": {"passed": True},
            "velocity": {"vel70": {"gate": {"passed": True, "failures": []}}},
        }
    }


def write_json(path, data):
    path.write_text(json.dumps(data))
    return path
