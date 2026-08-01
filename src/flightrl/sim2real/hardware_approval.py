from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn


class HardwareApprovalError(RuntimeError):
    pass


EDGE_BUNDLE_REQUIRED = (
    "generic checkpoint manifests cannot authorize learned live control; "
    "an exact typed edge-v3 deployment bundle is required"
)


def require_hardware_approved(
    checkpoint: str | Path,
    manifest: str | Path,
) -> NoReturn:
    del checkpoint
    manifest_path = Path(manifest)
    if not manifest_path.is_file():
        raise HardwareApprovalError(f"hardware approval manifest not found: {manifest_path}")
    raise HardwareApprovalError(EDGE_BUNDLE_REQUIRED)


def hardware_approval_status(
    checkpoint: str | Path,
    manifest: str | Path | None,
) -> dict[str, Any]:
    del checkpoint
    if manifest is None:
        return blocked_status("not_evaluated", "approval manifest not supplied", "")
    manifest_path = Path(manifest)
    if not manifest_path.is_file():
        return blocked_status("blocked", f"hardware approval manifest not found: {manifest_path}", manifest_path)
    return blocked_status("blocked", EDGE_BUNDLE_REQUIRED, manifest_path)


def blocked_status(
    status: str,
    error: str,
    manifest: str | Path,
) -> dict[str, Any]:
    return {
        "hardware_approved": False,
        "approval_status": status,
        "approval_error": error,
        "approval_manifest": str(manifest),
    }
