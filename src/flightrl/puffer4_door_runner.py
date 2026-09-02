from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
from typing import Sequence

import torch

from flightrl.artifact_identity import sha256_file, sha256_payload
from flightrl.puffer4_door_export import DOOR_NATIVE_FILES
from flightrl.puffer4_native_revision import require_clean_puffer_revision


BUILD_FINGERPRINT_SCHEMA_VERSION = 2
BUILD_MODE = "cpu"
PUFFER_NATIVE_FILES = (
    "build.sh",
    "src/bindings_cpu.cpp",
    "src/vecenv.h",
    "src/tensor.h",
)
_EXTENSION_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX")
_PYTHON_CACHE_TAG = sys.implementation.cache_tag


def current_python_abi() -> dict[str, str]:
    if not _EXTENSION_SUFFIX or not _PYTHON_CACHE_TAG:
        raise RuntimeError("Python does not expose a native extension ABI")
    return {
        "ext_suffix": str(_EXTENSION_SUFFIX),
        "cache_tag": str(_PYTHON_CACHE_TAG),
    }


def native_extension_path(puffer_root: Path) -> Path:
    root = Path(puffer_root).expanduser().resolve()
    return root / "pufferlib" / f"_C{current_python_abi()['ext_suffix']}"


def native_build_marker_path(puffer_root: Path) -> Path:
    extension = native_extension_path(puffer_root)
    return extension.with_name(extension.name + ".flightrl-build.json")


def native_source_paths(
    puffer_root: Path,
    env_name: str,
    native_files: Sequence[str] = DOOR_NATIVE_FILES,
) -> tuple[Path, ...]:
    root = Path(puffer_root).expanduser().resolve()
    env_dir = root / "ocean" / env_name
    relative_env_files = ("binding.c", *native_files)
    return tuple(
        sorted(
            (
                *(env_dir / name for name in relative_env_files),
                *(root / name for name in PUFFER_NATIVE_FILES),
            ),
            key=lambda path: str(path.resolve()),
        )
    )


def build_environment(
    puffer_root: Path,
    env_name: str,
    native_files: Sequence[str] = DOOR_NATIVE_FILES,
) -> dict:
    root = Path(puffer_root).expanduser().resolve()
    marker = native_build_marker_path(root)
    dependency_revision = require_clean_puffer_revision(root)
    before = _source_manifest(root, env_name, native_files)
    before_digest = _manifest_sha256(before)
    marker.unlink(missing_ok=True)
    env = os.environ.copy()
    python_bin = str(Path(sys.executable).resolve().parent)
    env["PATH"] = python_bin + os.pathsep + env.get("PATH", "")
    if sys.platform == "darwin":
        llvm = Path("/opt/homebrew/opt/llvm/bin")
        env.update({"CC": str(llvm / "clang"), "CXX": str(llvm / "clang++")})
    subprocess.run(
        ["bash", "build.sh", env_name, "--cpu"],
        cwd=root,
        env=env,
        check=True,
    )
    if (
        sys.platform == "darwin"
        and "/opt/homebrew/opt/llvm/lib"
        not in os.environ.get("DYLD_LIBRARY_PATH", "").split(":")
    ):
        _align_openmp_runtime(root)
    after = _source_manifest(root, env_name, native_files)
    after_digest = _manifest_sha256(after)
    if (
        before != after
        or before_digest != after_digest
        or dependency_revision != require_clean_puffer_revision(root)
    ):
        raise RuntimeError("native build sources changed during native build")
    extension = native_extension_path(root)
    if not extension.is_file():
        raise RuntimeError(f"native build did not produce extension: {extension}")
    fingerprint = {
        "schema_version": BUILD_FINGERPRINT_SCHEMA_VERSION,
        "env_name": env_name,
        "build_mode": BUILD_MODE,
        "python_abi": current_python_abi(),
        "dependency_revision": dependency_revision,
        "source_files_sha256": after,
        "source_manifest_sha256": after_digest,
        "source_manifest_sha256_before": before_digest,
        "source_manifest_sha256_after": after_digest,
        "extension": {
            "path": str(extension.resolve()),
            "sha256": _file_sha256(extension),
        },
    }
    _write_json_atomic(marker, fingerprint)
    return fingerprint


def _align_openmp_runtime(puffer_root: Path) -> None:
    torch_openmp = Path(torch.__file__).resolve().parent / "lib" / "libomp.dylib"
    extension = native_extension_path(puffer_root)
    dependencies = subprocess.run(
        ["otool", "-L", str(extension)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for line in dependencies.splitlines():
        dependency = line.strip().split(" ", 1)[0]
        if dependency.endswith("/libomp.dylib") and Path(dependency) != torch_openmp:
            subprocess.run(
                [
                    "install_name_tool",
                    "-change",
                    dependency,
                    str(torch_openmp),
                    str(extension),
                ],
                check=True,
            )


def verify_native_build(
    puffer_root: Path,
    env_name: str,
    native_files: Sequence[str] = DOOR_NATIVE_FILES,
) -> dict:
    root = Path(puffer_root).expanduser().resolve()
    marker = native_build_marker_path(root)
    if not marker.is_file():
        raise RuntimeError(
            f"native build fingerprint is missing: {marker}; rebuild without --skip-build"
        )
    try:
        fingerprint = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"native build fingerprint is unreadable: {marker}") from exc
    if fingerprint.get("schema_version") != BUILD_FINGERPRINT_SCHEMA_VERSION:
        raise RuntimeError("native build fingerprint schema is unsupported")
    if fingerprint.get("env_name") != env_name:
        raise RuntimeError(
            "native build fingerprint environment does not match "
            f"{env_name!r}"
        )
    if fingerprint.get("build_mode") != BUILD_MODE:
        raise RuntimeError("native build fingerprint is not for CPU mode")
    if fingerprint.get("python_abi") != current_python_abi():
        raise RuntimeError("native build fingerprint Python ABI does not match")
    if fingerprint.get("dependency_revision") != require_clean_puffer_revision(root):
        raise RuntimeError("native build PufferLib dependency revision does not match")
    expected_manifest = _source_manifest(root, env_name, native_files)
    expected_digest = _manifest_sha256(expected_manifest)
    recorded_digests = (
        fingerprint.get("source_manifest_sha256"),
        fingerprint.get("source_manifest_sha256_before"),
        fingerprint.get("source_manifest_sha256_after"),
    )
    if (
        fingerprint.get("source_files_sha256") != expected_manifest
        or any(item != expected_digest for item in recorded_digests)
    ):
        raise RuntimeError("native build source manifest does not match current sources")
    extension = native_extension_path(root)
    expected_extension = {
        "path": str(extension.resolve()),
        "sha256": _file_sha256(extension) if extension.is_file() else None,
    }
    if fingerprint.get("extension", {}).get("path") != expected_extension["path"]:
        raise RuntimeError("native build fingerprint extension path does not match")
    if fingerprint.get("extension", {}).get("sha256") != expected_extension["sha256"]:
        raise RuntimeError("native extension SHA-256 does not match its fingerprint")
    return fingerprint


def load_puffer(
    puffer_root: Path,
    env_name: str,
    native_files: Sequence[str] = DOOR_NATIVE_FILES,
):
    root = Path(puffer_root).expanduser().resolve()
    verify_native_build(root, env_name, native_files)
    preloaded = next(
        (
            name
            for name in sys.modules
            if name == "pufferlib" or name.startswith("pufferlib.")
        ),
        None,
    )
    if preloaded is not None:
        raise RuntimeError(
            f"{preloaded} is already loaded; restart before loading a "
            "fingerprinted native build"
        )
    root_text = str(root)
    sys.path[:] = [item for item in sys.path if item != root_text]
    sys.path.insert(0, root_text)
    from pufferlib import pufferl, torch_pufferl

    extension = native_extension_path(root)
    loaded_path = getattr(torch_pufferl._C, "__file__", None)
    if loaded_path is None or Path(loaded_path).resolve() != extension.resolve():
        raise RuntimeError(
            "pufferlib native extension was imported from the wrong path: "
            f"{loaded_path!r}"
        )
    compiled_env = getattr(torch_pufferl._C, "env_name", None)
    if compiled_env != env_name:
        raise RuntimeError(
            f"PufferLib native extension is built for {compiled_env!r}, "
            f"not {env_name!r}"
        )
    old_argv = sys.argv
    try:
        sys.argv = ["evaluate_puffer_fixed_door_teacher"]
        args = pufferl.load_config(env_name)
    finally:
        sys.argv = old_argv
    args["world_size"] = 1
    return args, torch_pufferl


def _source_manifest(
    puffer_root: Path,
    env_name: str,
    native_files: Sequence[str],
) -> dict[str, str]:
    return {
        str(path.resolve()): _file_sha256(path)
        for path in native_source_paths(puffer_root, env_name, native_files)
    }


def _manifest_sha256(manifest: dict[str, str]) -> str:
    return sha256_payload(manifest)


def _file_sha256(path: Path) -> str:
    return sha256_file(path)


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
