from __future__ import annotations

import hashlib

from flightrl.puffer4_door_provenance import (
    build_door_run_provenance,
    build_file_manifest,
)


def test_run_provenance_binds_command_runtime_and_exact_inputs(tmp_path) -> None:
    source_report = tmp_path / "parent.json"
    source_report.write_text('{"checkpoint_sha256":"abc"}\n')
    flight_source = tmp_path / "binding.c"
    flight_source.write_text("native-source\n")
    generated_config = tmp_path / "door.ini"
    generated_config.write_text("[env]\n")
    source_manifest = build_file_manifest(tmp_path, [flight_source])
    flight_source.write_text("changed-after-run-start\n")
    native_build = {
        "schema_version": 1,
        "env_name": "flightrl_fixed_door_d1",
        "extension": {"sha256": "native-extension"},
    }

    result = build_door_run_provenance(
        command=["python3.13", "train.py", "--seed", "11"],
        started_at_utc="2026-07-30T23:00:00+00:00",
        elapsed_wall_s=12.5,
        source_report=source_report,
        flightrl_root=tmp_path,
        flightrl_source_sha256=source_manifest,
        puffer_root=tmp_path,
        generated_files=[generated_config],
        native_build_fingerprint=native_build,
    )

    assert result["command"] == ["python3.13", "train.py", "--seed", "11"]
    assert result["elapsed_wall_s"] == 12.5
    assert result["source_report_sha256"] == hashlib.sha256(
        source_report.read_bytes()
    ).hexdigest()
    assert result["flightrl_source_sha256"]["binding.c"] == hashlib.sha256(
        b"native-source\n"
    ).hexdigest()
    assert result["generated_puffer_sha256"]["door.ini"]
    assert result["native_build_fingerprint"] == native_build
