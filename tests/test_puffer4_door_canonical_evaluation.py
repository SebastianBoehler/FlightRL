from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import flightrl.puffer4_door_canonical_evaluation as canonical
from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
)
from flightrl.puffer4_door_eval_provenance import (
    begin_fixed_door_evaluation_provenance,
    fixed_door_generated_paths,
)


def test_canonical_evaluation_module_exists() -> None:
    assert importlib.util.find_spec(
        "flightrl.puffer4_door_canonical_evaluation"
    ) is not None


class _ActionContract:
    max_yawrate_deg_s = 70.0

    @staticmethod
    def sha256() -> str:
        return "action-sha"


def _metrics(
    *,
    success: float = 0.82,
    outside_fov_success: float = 0.72,
    collision: float = 0.01,
) -> dict:
    return {
        "status": "complete",
        "success_rate": success,
        "outside_fov_success_rate": outside_fov_success,
        "collision_rate": collision,
        "finite_outputs": {"passed": True},
        "marginal_groups": {"worst_marginal_group": "outside_fov"},
    }


def test_canonical_runner_writes_bound_report_and_isolated_ablations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def fake_evaluate(_policy, _args, _puffer, **kwargs) -> dict:
        calls.append(kwargs)
        if kwargs["camera_mask"]:
            return _metrics(success=0.01, outside_fov_success=0.0)
        return _metrics()

    monkeypatch.setattr(
        canonical,
        "evaluate_promotion_door_policy",
        fake_evaluate,
    )
    output = tmp_path / "door.promotion-evaluation.json"
    lineage_report = tmp_path / "lineage.json"
    lineage_report.write_text('{"lineage": true}')
    root = Path(__file__).resolve().parents[1]
    capture = begin_fixed_door_evaluation_provenance(
        command=("python", "evaluate.py"),
        flightrl_root=root,
        entrypoint=root / "scripts/evaluate_puffer_fixed_door_checkpoint.py",
    )
    puffer_root = tmp_path / "Puffer"
    for path in fixed_door_generated_paths(
        puffer_root,
        "flightrl_fixed_door_d1",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
    trained_identity = {
        "checkpoint": {"path": "/tmp/door.bin", "sha256": "checkpoint-sha"},
        "action_contract": {"sha256": "action-sha"},
        "policy_contract": {"sha256": "policy-sha"},
        "environment": {"name": "flightrl_fixed_door_d1"},
        "train_seed": 11,
    }
    bundle = SimpleNamespace(
        action_contract=_ActionContract(),
        policy_contract={"sha256": "policy-sha"},
        report_path=lineage_report,
        env_name="flightrl_fixed_door_d1",
        trained_identity=lambda: trained_identity,
        lineage=lambda: {
            "report": {"path": str(lineage_report), "sha256": "lineage-sha"}
        },
    )

    report, written = canonical.run_canonical_door_evaluation(
        bundle=bundle,
        policy=object(),
        puffer_args={"env": {}, "vec": {}},
        torch_pufferl=object(),
        output=output,
        native_build_fingerprint={"extension_sha256": "native-sha"},
        stream_contract={"contract_id": "stream"},
        provenance_capture=capture,
        puffer_root=puffer_root,
        steps=3,
        seed=10_011,
        agents=2,
        live_yaw_cap_challenge=True,
    )

    assert written == output.resolve()
    assert json.loads(output.read_text()) == report
    identity = report["evaluation_identity"]
    assert identity["evidence_age_runtime_contract"] == (
        FIXED_DOOR_EVIDENCE_AGE_CONTRACT.to_report()
    )
    assert "evidence_age_runtime_contract" not in report["trained_identity"]
    assert report["simulation_gate"]["passed"] is True
    assert report["evaluation_provenance"]["source_report"] == str(
        lineage_report.resolve()
    )
    assert [call["camera_mask"] for call in calls] == [
        False,
        True,
        False,
        False,
        False,
    ]
    assert calls[2]["recurrent_mode"] == "reset_each_step"
    assert "temporal_order_seed" in calls[3]
    assert calls[4]["yaw_abs_limit_normalized"] == pytest.approx(8.0 / 70.0)
