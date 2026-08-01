from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
import hashlib
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal


ArtifactManifest = Sequence[tuple[str, str]]
ArtifactResolver = Callable[..., str | Path]
RuntimeVersions = Sequence[tuple[str, str]]
WeightsFormat = Literal["pytorch", "safetensors"]


def validate_huggingface_artifact_identity(
    *,
    model_id: str,
    revision: str,
    filename: str,
    sha256: str,
) -> None:
    if not model_id or "/" not in model_id:
        raise ValueError("Hugging Face model ID is invalid")
    if len(revision) != 40 or not _lower_hex(revision):
        raise ValueError("Hugging Face revision must be a full commit SHA")
    if not filename or Path(filename).name != filename:
        raise ValueError("Hugging Face artifact filename is invalid")
    if len(sha256) != 64 or not _lower_hex(sha256):
        raise ValueError("Hugging Face artifact SHA-256 is invalid")


def validate_optional_huggingface_snapshot(
    *,
    model_id: str,
    revision: str | None,
    manifest: ArtifactManifest,
    runtime_versions: RuntimeVersions,
) -> None:
    if revision is None:
        if manifest or runtime_versions:
            raise ValueError("Hugging Face snapshot identity is incomplete")
        return
    if not manifest:
        raise ValueError("Hugging Face snapshot manifest is empty")
    filenames = [filename for filename, _ in manifest]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Hugging Face snapshot manifest has duplicate files")
    for filename, sha256 in manifest:
        validate_huggingface_artifact_identity(
            model_id=model_id,
            revision=revision,
            filename=filename,
            sha256=sha256,
        )
    packages = [package for package, _ in runtime_versions]
    if (
        any(not package or not expected for package, expected in runtime_versions)
        or len(packages) != len(set(packages))
    ):
        raise ValueError("Hugging Face runtime identity is invalid")


def validate_huggingface_weights_format(
    *,
    revision: str | None,
    manifest: ArtifactManifest,
    weights_format: WeightsFormat | None,
) -> None:
    if revision is None:
        if weights_format is not None:
            raise ValueError("Hugging Face weights format requires a snapshot")
        return
    expected_filename = {
        "pytorch": "pytorch_model.bin",
        "safetensors": "model.safetensors",
    }.get(weights_format or "")
    if expected_filename is None:
        raise ValueError("Hugging Face snapshot weights format is invalid")
    if expected_filename not in {filename for filename, _ in manifest}:
        raise ValueError(
            "Hugging Face snapshot is missing its declared weight artifact"
        )


@contextmanager
def huggingface_model_source(
    *,
    model_id: str,
    revision: str | None,
    manifest: ArtifactManifest,
    runtime_versions: RuntimeVersions,
    resolver: ArtifactResolver | None = None,
) -> Iterator[str | Path]:
    """Yield either a generic model ID or an approved private local snapshot."""
    validate_optional_huggingface_snapshot(
        model_id=model_id,
        revision=revision,
        manifest=manifest,
        runtime_versions=runtime_versions,
    )
    if revision is None:
        yield model_id
        return
    with verified_local_huggingface_snapshot(
        model_id=model_id,
        revision=revision,
        manifest=manifest,
        runtime_versions=runtime_versions,
        resolver=resolver,
    ) as snapshot:
        yield snapshot


@contextmanager
def verified_local_huggingface_snapshot(
    *,
    model_id: str,
    revision: str,
    manifest: ArtifactManifest,
    runtime_versions: RuntimeVersions,
    resolver: ArtifactResolver | None = None,
) -> Iterator[Path]:
    """Copy every approved Hub file into a clean verified private snapshot."""
    validate_optional_huggingface_snapshot(
        model_id=model_id,
        revision=revision,
        manifest=manifest,
        runtime_versions=runtime_versions,
    )
    _require_runtime_versions(runtime_versions)
    if resolver is None:
        from huggingface_hub import hf_hub_download

        resolver = hf_hub_download
    with TemporaryDirectory(prefix="flightrl-hf-snapshot-") as temporary:
        snapshot = Path(temporary)
        for filename, expected_sha256 in sorted(manifest):
            _copy_verified_artifact(
                snapshot=snapshot,
                model_id=model_id,
                revision=revision,
                filename=filename,
                expected_sha256=expected_sha256,
                resolver=resolver,
            )
        snapshot.chmod(0o500)
        yield snapshot


def _copy_verified_artifact(
    *,
    snapshot: Path,
    model_id: str,
    revision: str,
    filename: str,
    expected_sha256: str,
    resolver: ArtifactResolver,
) -> None:
    try:
        source = Path(
            resolver(
                repo_id=model_id,
                revision=revision,
                filename=filename,
                local_files_only=True,
            )
        )
        destination = snapshot / filename
        digest = hashlib.sha256()
        with source.open("rb") as source_handle:
            with destination.open("xb") as destination_handle:
                for chunk in iter(
                    lambda: source_handle.read(1024 * 1024),
                    b"",
                ):
                    digest.update(chunk)
                    destination_handle.write(chunk)
    except OSError as exc:
        raise ValueError("approved Hugging Face artifact is unavailable") from exc
    if digest.hexdigest() != expected_sha256:
        raise ValueError("approved Hugging Face artifact SHA-256 does not match")
    destination.chmod(0o400)


def _require_runtime_versions(runtime_versions: RuntimeVersions) -> None:
    for package, expected in runtime_versions:
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise ValueError(
                f"approved Hugging Face runtime package is missing: {package}"
            ) from exc
        if actual != expected:
            raise ValueError(
                "approved Hugging Face runtime version does not match: "
                f"{package}=={actual}, expected {expected}"
            )


def _lower_hex(value: str) -> bool:
    return all(character in "0123456789abcdef" for character in value)
