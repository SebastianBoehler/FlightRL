from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from flightrl.puffer4_door_runner import (
    BUILD_FINGERPRINT_SCHEMA_VERSION,
    BUILD_MODE,
    current_python_abi,
)
from flightrl.puffer4_edge_native_build import (
    canonical_edge_native_build_fingerprint,
    require_matching_edge_native_build_fingerprints,
)


def _fingerprint(root: Path) -> dict:
    source = str((root / "ocean/edge/binding.c").resolve())
    manifest = {source: "a" * 64}
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": BUILD_FINGERPRINT_SCHEMA_VERSION,
        "env_name": "edge",
        "build_mode": BUILD_MODE,
        "python_abi": current_python_abi(),
        "dependency_revision": {"git_commit": "b" * 40},
        "source_files_sha256": manifest,
        "source_manifest_sha256": digest,
        "source_manifest_sha256_before": digest,
        "source_manifest_sha256_after": digest,
        "extension": {
            "path": str((root / "pufferlib/_C.test.so").resolve()),
            "sha256": "c" * 64,
        },
    }


def test_edge_native_build_fingerprint_is_canonical_and_exact(tmp_path: Path) -> None:
    fingerprint = _fingerprint(tmp_path)

    assert canonical_edge_native_build_fingerprint(fingerprint) == fingerprint
    assert require_matching_edge_native_build_fingerprints(
        fingerprint,
        deepcopy(fingerprint),
    ) == fingerprint

    changed = deepcopy(fingerprint)
    changed["extension"]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="do not match"):
        require_matching_edge_native_build_fingerprints(fingerprint, changed)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("legacy", "schema"),
        ("revision", "revision"),
        ("manifest", "manifest"),
        ("relative_extension", "absolute"),
        ("abi", "Python ABI"),
    ],
)
def test_edge_native_build_fingerprint_rejects_noncanonical_identity(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    fingerprint = _fingerprint(tmp_path)
    if mutation == "legacy":
        fingerprint["schema_version"] = 1
    elif mutation == "revision":
        fingerprint["dependency_revision"]["git_commit"] = "not-a-commit"
    elif mutation == "manifest":
        fingerprint["source_files_sha256"][next(iter(fingerprint["source_files_sha256"]))] = "d" * 64
    elif mutation == "relative_extension":
        fingerprint["extension"]["path"] = "pufferlib/_C.test.so"
    else:
        fingerprint["python_abi"] = {"cache_tag": "cpython-test"}

    with pytest.raises(ValueError, match=match):
        canonical_edge_native_build_fingerprint(fingerprint)
