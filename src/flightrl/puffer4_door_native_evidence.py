from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from flightrl.puffer4_door_runner import native_source_paths


def validate_native_build_fingerprint(
    fingerprint: Mapping[str, Any],
    env_name: str,
) -> None:
    if (
        fingerprint.get("schema_version") != 1
        or fingerprint.get("env_name") != env_name
        or fingerprint.get("build_mode") != "cpu"
    ):
        raise ValueError("native build fingerprint target is invalid")
    abi = _mapping(fingerprint.get("python_abi"), "native Python ABI")
    suffix = abi.get("ext_suffix")
    cache_tag = abi.get("cache_tag")
    if not isinstance(suffix, str) or not suffix or not isinstance(cache_tag, str):
        raise ValueError("native build fingerprint Python ABI is incomplete")

    extension = _mapping(
        fingerprint.get("extension"),
        "native build extension identity",
    )
    encoded_extension = extension.get("path")
    if not isinstance(encoded_extension, str):
        raise ValueError("native build extension path is missing")
    extension_path = Path(encoded_extension)
    if not extension_path.is_absolute():
        raise ValueError("native build extension path must be absolute")
    puffer_root = extension_path.resolve().parent.parent
    expected_extension = puffer_root / "pufferlib" / f"_C{suffix}"
    if extension_path.resolve() != expected_extension:
        raise ValueError("native build extension path does not match its ABI")
    _require_sha256(extension.get("sha256"), "native extension")

    manifest = _mapping(
        fingerprint.get("source_files_sha256"),
        "native source manifest",
    )
    expected_paths = {
        str(path.resolve()) for path in native_source_paths(puffer_root, env_name)
    }
    if set(manifest) != expected_paths:
        raise ValueError("native source manifest does not contain exact build inputs")
    for path, digest in manifest.items():
        if not Path(path).is_absolute():
            raise ValueError("native source manifest paths must be absolute")
        _require_sha256(digest, f"native source {path}")
    manifest_digest = hashlib.sha256(
        json.dumps(
            dict(manifest),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    recorded = (
        fingerprint.get("source_manifest_sha256"),
        fingerprint.get("source_manifest_sha256_before"),
        fingerprint.get("source_manifest_sha256_after"),
    )
    if any(value != manifest_digest for value in recorded):
        raise ValueError("native source manifest digest is inconsistent")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is missing or invalid")
    return value


def _require_sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} SHA-256 is invalid")
