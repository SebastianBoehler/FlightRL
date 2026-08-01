from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from flightrl.puffer4_door_eval_provenance import (
    begin_fixed_door_evaluation_provenance,
    fixed_door_evaluation_source_paths,
    fixed_door_generated_paths,
    write_fixed_door_evaluation_report,
)


def test_evaluation_source_paths_bind_entrypoint_and_door_runtime(
    tmp_path: Path,
) -> None:
    entrypoint = tmp_path / "scripts" / "evaluate.py"
    door_python = tmp_path / "src" / "flightrl" / "puffer4_door_policy.py"
    door_native = (
        tmp_path / "src" / "flightrl" / "native" / "native_door_action.c"
    )
    sixdof_native = (
        tmp_path
        / "src"
        / "flightrl"
        / "native"
        / "native_sixdof_setpoint.c"
    )
    binding = (
        tmp_path
        / "src"
        / "flightrl"
        / "native"
        / "native_door_env_binding.c"
    )
    unrelated = tmp_path / "src" / "flightrl" / "unrelated.py"
    for path in (
        entrypoint,
        door_python,
        door_native,
        sixdof_native,
        binding,
        unrelated,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name)

    paths = fixed_door_evaluation_source_paths(tmp_path, entrypoint)

    assert set(paths) == {
        entrypoint,
        door_python,
        door_native,
        sixdof_native,
        binding,
    }


def _provenance_fixture(tmp_path: Path):
    root = tmp_path / "FlightRL"
    entrypoint = root / "scripts" / "evaluate.py"
    door_source = root / "src" / "flightrl" / "puffer4_door_policy.py"
    for path in (entrypoint, door_source):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name)
    puffer_root = tmp_path / "Puffer"
    generated = fixed_door_generated_paths(puffer_root, "door_env")
    for path in generated:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
    lineage = tmp_path / "lineage.json"
    lineage.write_text('{"lineage": true}')
    return root, entrypoint, door_source, puffer_root, lineage


def test_writer_binds_complete_runtime_provenance_before_exclusive_write(
    tmp_path: Path,
) -> None:
    root, entrypoint, _, puffer_root, lineage = _provenance_fixture(tmp_path)
    capture = begin_fixed_door_evaluation_provenance(
        command=("python", "evaluate.py", "--seed", "31"),
        flightrl_root=root,
        entrypoint=entrypoint,
    )
    native = {"extension": {"sha256": "native-sha"}}
    output = tmp_path / "evaluation.json"

    report = write_fixed_door_evaluation_report(
        report={"evaluation_schema": "test"},
        output=output,
        capture=capture,
        lineage_report=lineage,
        puffer_root=puffer_root,
        env_name="door_env",
        native_build_fingerprint=native,
    )

    provenance = report["evaluation_provenance"]
    assert json.loads(output.read_text()) == report
    assert provenance["command"] == [
        "python",
        "evaluate.py",
        "--seed",
        "31",
    ]
    assert provenance["torch_version"] == torch.__version__
    assert provenance["elapsed_wall_s"] >= 0.0
    assert provenance["started_at_utc"]
    assert provenance["finished_at_utc"]
    assert provenance["source_report"] == str(lineage.resolve())
    assert provenance["source_report_sha256"] == hashlib.sha256(
        lineage.read_bytes()
    ).hexdigest()
    assert provenance["native_build_fingerprint"] == native
    assert "pufferlib/torch_pufferl.py" in provenance[
        "generated_puffer_sha256"
    ]


def test_writer_refuses_source_drift_without_creating_output(
    tmp_path: Path,
) -> None:
    root, entrypoint, door_source, puffer_root, lineage = _provenance_fixture(
        tmp_path
    )
    capture = begin_fixed_door_evaluation_provenance(
        command=("python", "evaluate.py"),
        flightrl_root=root,
        entrypoint=entrypoint,
    )
    door_source.write_text("changed during evaluation")
    output = tmp_path / "evaluation.json"

    with pytest.raises(RuntimeError, match="source manifest changed"):
        write_fixed_door_evaluation_report(
            report={},
            output=output,
            capture=capture,
            lineage_report=lineage,
            puffer_root=puffer_root,
            env_name="door_env",
            native_build_fingerprint={},
        )

    assert not output.exists()
