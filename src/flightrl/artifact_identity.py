"""Canonical identities for immutable FlightRL artifacts and contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def canonical_json_bytes(value: object) -> bytes:
    """Encode identity-bearing JSON without whitespace or non-finite numbers."""
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON") from exc
    return encoded.encode("utf-8")


def sha256_payload(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def bind_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "sha256" in payload:
        raise ValueError("payload cannot contain a sha256 field before binding")
    result = dict(payload)
    result["sha256"] = sha256_payload(result)
    return result


def require_bound_payload(
    report: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    payload = {key: value for key, value in report.items() if key != "sha256"}
    if report.get("sha256") != sha256_payload(payload):
        raise ValueError(f"{label} SHA-256 does not match")
    return payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}
