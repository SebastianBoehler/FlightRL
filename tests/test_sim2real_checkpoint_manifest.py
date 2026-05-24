from __future__ import annotations

import json
import subprocess
import sys

from flightrl.sim2real.checkpoint_manifest import build_checkpoint_manifest


def test_manifest_marks_sim_ready_checkpoints_as_sim_only_when_transfer_blocked(tmp_path) -> None:
    gate = write_json(tmp_path / "gate.json", {"transfer_approved": False})
    sim = write_json(tmp_path / "sim.json", readiness([record("circle", ready=True)]))
    deploy = write_json(tmp_path / "deploy.json", readiness([record("circle", ready=False, failures=["replay_comparison"])]))

    report = build_checkpoint_manifest(transfer_gate=gate, sim_readiness=sim, deployment_readiness=deploy)

    assert report["records"][0]["status"] == "sim_only"
    assert report["records"][0]["hardware_approved"] is False
    assert report["summary"]["sim_ready"] == 1
    assert report["summary"]["hardware_approved"] == 0
    assert report["inputs"]["transfer_gate"]["sha256"]


def test_manifest_approves_checkpoint_only_when_gate_and_deployment_pass(tmp_path) -> None:
    gate = write_json(tmp_path / "gate.json", {"transfer_approved": True})
    sim = write_json(tmp_path / "sim.json", readiness([record("multitask", ready=True)]))
    deploy = write_json(tmp_path / "deploy.json", readiness([record("multitask", ready=True)]))

    report = build_checkpoint_manifest(transfer_gate=gate, sim_readiness=sim, deployment_readiness=deploy)

    assert report["records"][0]["status"] == "hardware_approved"
    assert report["summary"]["hardware_approved"] == 1


def test_manifest_cli_writes_report(tmp_path) -> None:
    gate = write_json(tmp_path / "gate.json", {"transfer_approved": False})
    sim = write_json(tmp_path / "sim.json", readiness([record("position_yaw", ready=True)]))
    deploy = write_json(tmp_path / "deploy.json", readiness([record("position_yaw", ready=False, failures=["replay_comparison"])]))
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


def readiness(records):
    ready = [item for item in records if item["ready"]]
    return {"summary": {"total": len(records), "ready": len(ready), "blocked": len(records) - len(ready)}, "records": records}


def record(task: str, *, ready: bool, failures: list[str] | None = None):
    return {
        "task": task,
        "label": f"{task}_label",
        "checkpoint": f"artifacts/checkpoints/{task}.pt",
        "tasks": [task],
        "ready": ready,
        "failures": failures or [],
    }


def write_json(path, data):
    path.write_text(json.dumps(data))
    return path
