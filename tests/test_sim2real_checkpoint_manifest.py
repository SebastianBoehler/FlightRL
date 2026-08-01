from __future__ import annotations

import json
import subprocess
import sys

from flightrl.evidence_scope import EDGE_DEPLOYMENT_VERIFIER_MISSING
from flightrl.sim2real.checkpoint_manifest import build_checkpoint_manifest


def test_manifest_marks_sim_ready_checkpoints_as_sim_only_when_transfer_blocked(tmp_path) -> None:
    gate = write_json(tmp_path / "gate.json", {"transfer_approved": False})
    sim = write_json(tmp_path / "sim.json", readiness([record("circle", ready=True)]))
    deploy = write_json(tmp_path / "deploy.json", readiness([record("circle", ready=False, failures=["replay_comparison"])], deployment=True))

    report = build_checkpoint_manifest(transfer_gate=gate, sim_readiness=sim, deployment_readiness=deploy)

    assert report["records"][0]["status"] == "sim_only"
    assert report["records"][0]["hardware_approved"] is False
    assert report["summary"]["sim_ready"] == 1
    assert report["summary"]["hardware_approved"] == 0
    assert report["inputs"]["transfer_gate"]["sha256"]


def test_manifest_cannot_promote_without_typed_edge_bundle_verifier(tmp_path) -> None:
    gate = write_json(tmp_path / "gate.json", {"transfer_approved": True})
    checkpoint = tmp_path / "multitask.pt"
    checkpoint.write_text("checkpoint")
    sim = write_json(tmp_path / "sim.json", readiness([record("multitask", ready=True, checkpoint=checkpoint)]))
    deploy = write_json(tmp_path / "deploy.json", readiness([record("multitask", ready=True, checkpoint=checkpoint)], deployment=True))

    report = build_checkpoint_manifest(transfer_gate=gate, sim_readiness=sim, deployment_readiness=deploy)

    assert report["deployment_authority"] is False
    assert report["transfer_approved"] is False
    assert report["records"][0]["deployment_ready"] is False
    assert report["records"][0]["status"] == "sim_only"
    assert EDGE_DEPLOYMENT_VERIFIER_MISSING in report["records"][0]["deployment_failures"]
    assert report["records"][0]["hardware_blocker"] == "edge_v3_deployment_bundle_required"
    assert report["summary"]["hardware_approved"] == 0
    assert report["records"][0]["checkpoint_provenance"]["sha256"]


def test_manifest_never_approves_a_missing_checkpoint(tmp_path) -> None:
    gate = write_json(tmp_path / "gate.json", {"transfer_approved": True})
    missing = tmp_path / "missing.pt"
    sim = write_json(tmp_path / "sim.json", readiness([record("circle", ready=True, checkpoint=missing)]))
    deploy = write_json(tmp_path / "deploy.json", readiness([record("circle", ready=True, checkpoint=missing)], deployment=True))

    report = build_checkpoint_manifest(
        transfer_gate=gate,
        sim_readiness=sim,
        deployment_readiness=deploy,
    )

    assert report["summary"]["hardware_approved"] == 0
    assert EDGE_DEPLOYMENT_VERIFIER_MISSING in report["records"][0]["deployment_failures"]
    assert "checkpoint_missing" in report["records"][0]["deployment_failures"]


def test_manifest_never_approves_when_simulation_is_not_ready(tmp_path) -> None:
    gate = write_json(tmp_path / "gate.json", {"transfer_approved": True})
    checkpoint = tmp_path / "circle.pt"
    checkpoint.write_text("checkpoint")
    sim = write_json(
        tmp_path / "sim.json",
        readiness([record("circle", ready=False, failures=["held_out_success"], checkpoint=checkpoint)]),
    )
    deploy = write_json(
        tmp_path / "deploy.json",
        readiness([record("circle", ready=True, checkpoint=checkpoint)], deployment=True),
    )

    report = build_checkpoint_manifest(
        transfer_gate=gate,
        sim_readiness=sim,
        deployment_readiness=deploy,
    )

    assert report["records"][0]["deployment_ready"] is False
    assert report["records"][0]["hardware_approved"] is False
    assert report["records"][0]["status"] == "blocked"
    assert EDGE_DEPLOYMENT_VERIFIER_MISSING in report["records"][0]["deployment_failures"]


def test_manifest_never_combines_evidence_from_different_checkpoints(tmp_path) -> None:
    gate = write_json(tmp_path / "gate.json", {"transfer_approved": True})
    sim_checkpoint = tmp_path / "sim.pt"
    deploy_checkpoint = tmp_path / "deploy.pt"
    sim_checkpoint.write_text("sim checkpoint")
    deploy_checkpoint.write_text("deploy checkpoint")
    sim = write_json(
        tmp_path / "sim.json",
        readiness([record("circle", ready=True, checkpoint=sim_checkpoint)]),
    )
    deploy = write_json(
        tmp_path / "deploy.json",
        readiness([record("circle", ready=True, checkpoint=deploy_checkpoint)], deployment=True),
    )

    report = build_checkpoint_manifest(
        transfer_gate=gate,
        sim_readiness=sim,
        deployment_readiness=deploy,
    )

    assert report["records"][0]["hardware_approved"] is False
    assert report["records"][0]["deployment_ready"] is False
    assert "checkpoint_identity_mismatch" in report["records"][0]["deployment_failures"]


def test_manifest_rejects_truthy_strings_and_ready_records_with_failures(tmp_path) -> None:
    checkpoint = tmp_path / "circle.pt"
    checkpoint.write_text("checkpoint")
    gate = write_json(tmp_path / "gate.json", {"transfer_approved": "false"})
    sim_record = record("circle", ready="false", checkpoint=checkpoint)
    deploy_record = record("circle", ready=True, failures=["deployment_failed"], checkpoint=checkpoint)
    sim = write_json(tmp_path / "sim.json", readiness([sim_record]))
    deploy = write_json(tmp_path / "deploy.json", readiness([deploy_record], deployment=True))

    report = build_checkpoint_manifest(
        transfer_gate=gate,
        sim_readiness=sim,
        deployment_readiness=deploy,
    )

    assert report["transfer_approved"] is False
    assert report["records"][0]["sim_ready"] is False
    assert "invalid_ready" in report["records"][0]["sim_failures"]
    assert report["records"][0]["deployment_ready"] is False
    assert report["summary"]["hardware_approved"] == 0


def test_manifest_rejects_integer_ready_value(tmp_path) -> None:
    checkpoint = tmp_path / "circle.pt"
    checkpoint.write_text("checkpoint")
    gate = write_json(tmp_path / "gate.json", {"transfer_approved": True})
    sim = write_json(tmp_path / "sim.json", readiness([record("circle", ready=1, checkpoint=checkpoint)]))
    deploy = write_json(
        tmp_path / "deploy.json",
        readiness([record("circle", ready=True, checkpoint=checkpoint)], deployment=True),
    )

    report = build_checkpoint_manifest(
        transfer_gate=gate,
        sim_readiness=sim,
        deployment_readiness=deploy,
    )

    assert report["records"][0]["sim_ready"] is False
    assert "invalid_ready" in report["records"][0]["sim_failures"]


def test_manifest_cli_writes_report(tmp_path) -> None:
    gate = write_json(tmp_path / "gate.json", {"transfer_approved": False})
    sim = write_json(tmp_path / "sim.json", readiness([record("position_yaw", ready=True)]))
    deploy = write_json(tmp_path / "deploy.json", readiness([record("position_yaw", ready=False, failures=["replay_comparison"])], deployment=True))
    output = tmp_path / "manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_sim2real_checkpoint_manifest.py",
            "--transfer-gate",
            str(gate),
            "--sim-readiness",
            str(sim),
            "--deployment-readiness",
            str(deploy),
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
    saved = json.loads(output.read_text())
    assert saved["inputs"]["sim_readiness"]["path"] == str(sim)


def readiness(records, *, deployment: bool = False):
    ready = [item for item in records if item["ready"]]
    scope = {"evidence_scope": "edge_deployment", "deployment_authority": True} if deployment else {}
    return {**scope, "summary": {"total": len(records), "ready": len(ready), "blocked": len(records) - len(ready)}, "records": records}


def record(task: str, *, ready, failures: list[str] | None = None, checkpoint=None):
    return {
        "task": task,
        "label": f"{task}_label",
        "checkpoint": str(checkpoint or f"artifacts/checkpoints/{task}.pt"),
        "tasks": [task],
        "ready": ready,
        "failures": failures or [],
    }


def write_json(path, data):
    path.write_text(json.dumps(data))
    return path
