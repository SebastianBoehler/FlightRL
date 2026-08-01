from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re

from flightrl.puffer4_door_runner import (
    BUILD_FINGERPRINT_SCHEMA_VERSION,
    BUILD_MODE,
    verify_native_build,
)
from flightrl.puffer4_edge_student_export import EDGE_STUDENT_NATIVE_FILES


_FIELDS = {
    "schema_version",
    "env_name",
    "build_mode",
    "python_abi",
    "dependency_revision",
    "source_files_sha256",
    "source_manifest_sha256",
    "source_manifest_sha256_before",
    "source_manifest_sha256_after",
    "extension",
}


def canonical_edge_native_build_fingerprint(value: object) -> dict:
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise ValueError("edge native build fingerprint fields are invalid")
    if type(value["schema_version"]) is not int or (
        value["schema_version"] != BUILD_FINGERPRINT_SCHEMA_VERSION
    ):
        raise ValueError("edge native build fingerprint schema is incompatible")
    env_name = value["env_name"]
    if not isinstance(env_name, str) or not env_name:
        raise ValueError("edge native build environment is invalid")
    if value["build_mode"] != BUILD_MODE:
        raise ValueError("edge native build mode is incompatible")
    python_abi = _string_mapping(
        value["python_abi"],
        {"ext_suffix", "cache_tag"},
        "Python ABI",
    )
    revision = _string_mapping(
        value["dependency_revision"],
        {"git_commit"},
        "dependency revision",
    )
    if re.fullmatch(r"[0-9a-f]{40}", revision["git_commit"]) is None:
        raise ValueError("edge native build dependency revision is invalid")
    sources = _source_manifest(value["source_files_sha256"])
    manifest_sha256 = hashlib.sha256(
        json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    for name in (
        "source_manifest_sha256",
        "source_manifest_sha256_before",
        "source_manifest_sha256_after",
    ):
        if value[name] != manifest_sha256:
            raise ValueError("edge native build source manifest does not reproduce")
    extension = value["extension"]
    if not isinstance(extension, Mapping) or set(extension) != {"path", "sha256"}:
        raise ValueError("edge native build extension identity is invalid")
    extension_path = _absolute_path(extension["path"], "extension")
    extension_sha256 = _digest(extension["sha256"], "extension")
    return {
        "schema_version": BUILD_FINGERPRINT_SCHEMA_VERSION,
        "env_name": env_name,
        "build_mode": BUILD_MODE,
        "python_abi": python_abi,
        "dependency_revision": revision,
        "source_files_sha256": sources,
        "source_manifest_sha256": manifest_sha256,
        "source_manifest_sha256_before": manifest_sha256,
        "source_manifest_sha256_after": manifest_sha256,
        "extension": {"path": extension_path, "sha256": extension_sha256},
    }


def require_matching_edge_native_build_fingerprints(*values: object) -> dict:
    if not values:
        raise ValueError("at least one edge native build fingerprint is required")
    fingerprints = [canonical_edge_native_build_fingerprint(value) for value in values]
    if any(value != fingerprints[0] for value in fingerprints[1:]):
        raise ValueError("edge native build fingerprints do not match exactly")
    return fingerprints[0]


def require_current_edge_native_build_fingerprint(
    value: object,
    *,
    expected: object | None = None,
) -> dict:
    fingerprint = canonical_edge_native_build_fingerprint(value)
    extension = Path(fingerprint["extension"]["path"])
    root = extension.parent.parent
    try:
        verified = verify_native_build(
            root,
            fingerprint["env_name"],
            EDGE_STUDENT_NATIVE_FILES,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"edge native build is invalid: {exc}") from exc
    current = require_matching_edge_native_build_fingerprints(fingerprint, verified)
    if expected is not None:
        current = require_matching_edge_native_build_fingerprints(current, expected)
    return current


def _source_manifest(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("edge native build source manifest is invalid")
    result = {}
    for path, digest in value.items():
        canonical_path = _absolute_path(path, "source")
        result[canonical_path] = _digest(digest, "source")
    if len(result) != len(value):
        raise ValueError("edge native build source paths are not unique")
    return result


def _string_mapping(value: object, fields: set[str], label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"edge native build {label} fields are invalid")
    result = dict(value)
    if any(not isinstance(item, str) or not item for item in result.values()):
        raise ValueError(f"edge native build {label} is invalid")
    return result


def _absolute_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"edge native build {label} path is invalid")
    path = Path(value)
    if not path.is_absolute() or str(path.resolve()) != value:
        raise ValueError(f"edge native build {label} path must be canonical and absolute")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"edge native build {label} SHA-256 is invalid")
    return value
