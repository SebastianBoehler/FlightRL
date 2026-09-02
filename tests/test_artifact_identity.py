from __future__ import annotations

from pathlib import Path

import pytest

from flightrl.artifact_identity import (
    bind_payload,
    canonical_json_bytes,
    file_identity,
    require_bound_payload,
    sha256_payload,
)


def test_canonical_payload_identity_is_stable() -> None:
    payload = {"z": 1, "a": "ä"}

    assert canonical_json_bytes(payload) == b'{"a":"\\u00e4","z":1}'
    assert sha256_payload(payload) == (
        "aa3d92a62a8c20f1c6320ae03680d2d8e6f3a19674f5aa550f2d28530bb1c558"
    )


def test_bound_payload_rejects_nonfinite_and_mutated_data() -> None:
    with pytest.raises(ValueError, match="canonical JSON"):
        bind_payload({"gain": float("nan")})

    report = bind_payload({"contract_id": "example-v1", "gain": 0.5})
    report["gain"] = 0.75

    with pytest.raises(ValueError, match="SHA-256"):
        require_bound_payload(report, label="example contract")


def test_file_identity_uses_canonical_path_and_content(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"FlightRL\n")

    assert file_identity(artifact) == {
        "path": str(artifact.resolve()),
        "sha256": "c0165b49e73c28c451889d1cfec31f8435133a3d05d87494a70599de795cd67f",
    }
