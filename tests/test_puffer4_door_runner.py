from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from flightrl.puffer4_door_export import DOOR_NATIVE_FILES
import flightrl.puffer4_door_runner as runner


ENV_NAME = "flightrl_fixed_door_test"


@pytest.fixture(autouse=True)
def _stable_dependency_revision(monkeypatch):
    monkeypatch.setattr(
        runner,
        "require_clean_puffer_revision",
        lambda _root: {"git_commit": "a" * 40},
    )


def _write_build_inputs(root: Path) -> Path:
    env_dir = root / "ocean" / ENV_NAME
    env_dir.mkdir(parents=True)
    (root / "pufferlib").mkdir()
    (root / "src").mkdir()
    for relative in (
        "build.sh",
        "src/bindings_cpu.cpp",
        "src/vecenv.h",
        "src/tensor.h",
    ):
        path = root / relative
        path.write_text(f"{relative}\n")
    (env_dir / "binding.c").write_text("binding\n")
    for name in DOOR_NATIVE_FILES:
        (env_dir / name).write_text(f"{name}\n")
    config = root / "config" / f"{ENV_NAME}.ini"
    config.parent.mkdir()
    config.write_text("[env]\nseed = 11\n")
    return env_dir


def _fake_successful_build(
    root: Path,
    extension_bytes: bytes | None = b"extension",
):
    def run(command, *, cwd, env, check):
        assert command == ["bash", "build.sh", ENV_NAME, "--cpu"]
        assert Path(cwd) == root
        assert Path(env["PATH"].split(":", 1)[0]) == Path(sys.executable).resolve().parent
        if extension_bytes is not None:
            runner.native_extension_path(root).write_bytes(extension_bytes)

    return run


@contextmanager
def _clean_puffer_modules():
    previous = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "pufferlib" or name.startswith("pufferlib.")
    }
    for name in previous:
        sys.modules.pop(name)
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if name == "pufferlib" or name.startswith("pufferlib."):
                sys.modules.pop(name)
        sys.modules.update(previous)


def test_current_python_abi_is_stable_when_build_target_platform_is_mocked(
    monkeypatch,
) -> None:
    expected = runner.current_python_abi()

    monkeypatch.setattr(runner.sys, "platform", "linux")

    assert runner.current_python_abi() == expected


def test_build_records_atomic_abi_specific_fingerprint_for_exact_sources(
    monkeypatch,
    tmp_path: Path,
) -> None:
    env_dir = _write_build_inputs(tmp_path)
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        _fake_successful_build(tmp_path),
    )
    replacements: list[tuple[Path, Path]] = []
    real_replace = runner.os.replace

    def record_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", record_replace)

    result = runner.build_environment(tmp_path, ENV_NAME)

    extension = runner.native_extension_path(tmp_path)
    marker = runner.native_build_marker_path(tmp_path)
    expected_sources = {
        *(env_dir / name for name in ("binding.c", *DOOR_NATIVE_FILES)),
        tmp_path / "build.sh",
        tmp_path / "src/bindings_cpu.cpp",
        tmp_path / "src/vecenv.h",
        tmp_path / "src/tensor.h",
    }
    assert marker.name == extension.name + ".flightrl-build.json"
    assert result == json.loads(marker.read_text())
    assert result["schema_version"] == 2
    assert result["env_name"] == ENV_NAME
    assert result["build_mode"] == "cpu"
    assert result["python_abi"] == runner.current_python_abi()
    assert result["dependency_revision"] == {"git_commit": "a" * 40}
    assert set(result["source_files_sha256"]) == {
        str(path.resolve()) for path in expected_sources
    }
    assert not any(path.endswith(".ini") for path in result["source_files_sha256"])
    assert result["extension"] == {
        "path": str(extension.resolve()),
        "sha256": hashlib.sha256(b"extension").hexdigest(),
    }
    assert (
        result["source_manifest_sha256_before"]
        == result["source_manifest_sha256_after"]
        == result["source_manifest_sha256"]
    )
    assert len(replacements) == 1
    assert replacements[0][1] == marker
    assert replacements[0][0].parent == marker.parent
    assert replacements[0][0] != marker
    assert not replacements[0][0].exists()


def test_build_rejects_source_change_and_removes_stale_marker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    env_dir = _write_build_inputs(tmp_path)
    marker = runner.native_build_marker_path(tmp_path)
    marker.write_text('{"stale":true}\n')
    monkeypatch.setattr(runner.sys, "platform", "linux")

    def mutate_during_build(command, *, cwd, env, check):
        (env_dir / "binding.c").write_text("changed-during-build\n")
        runner.native_extension_path(tmp_path).write_bytes(b"extension")

    monkeypatch.setattr(runner.subprocess, "run", mutate_during_build)

    with pytest.raises(RuntimeError, match="changed during native build"):
        runner.build_environment(tmp_path, ENV_NAME)

    assert not marker.exists()


def test_verifier_rejects_source_changed_after_build(monkeypatch, tmp_path: Path) -> None:
    env_dir = _write_build_inputs(tmp_path)
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.subprocess, "run", _fake_successful_build(tmp_path))
    runner.build_environment(tmp_path, ENV_NAME)
    (env_dir / "native_door_action.c").write_text("stale-source\n")

    with pytest.raises(RuntimeError, match="source manifest"):
        runner.verify_native_build(tmp_path, ENV_NAME)


def test_verifier_rejects_extension_changed_after_build(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_build_inputs(tmp_path)
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.subprocess, "run", _fake_successful_build(tmp_path))
    runner.build_environment(tmp_path, ENV_NAME)
    runner.native_extension_path(tmp_path).write_bytes(b"replaced-extension")

    with pytest.raises(RuntimeError, match="extension SHA-256"):
        runner.verify_native_build(tmp_path, ENV_NAME)


def test_verifier_rejects_marker_for_another_python_abi(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_build_inputs(tmp_path)
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.subprocess, "run", _fake_successful_build(tmp_path))
    runner.build_environment(tmp_path, ENV_NAME)
    marker = runner.native_build_marker_path(tmp_path)
    payload = json.loads(marker.read_text())
    payload["python_abi"]["cache_tag"] = "cpython-wrong"
    marker.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="Python ABI"):
        runner.verify_native_build(tmp_path, ENV_NAME)


def test_load_fails_before_import_when_marker_is_missing(tmp_path: Path) -> None:
    _write_build_inputs(tmp_path)
    sentinel = tmp_path / "imported"
    (tmp_path / "pufferlib" / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('yes')\n"
    )

    with _clean_puffer_modules():
        with pytest.raises(RuntimeError, match="fingerprint is missing"):
            runner.load_puffer(tmp_path, ENV_NAME)

    assert not sentinel.exists()


@pytest.mark.parametrize(
    "module_name",
    ("pufferlib", "pufferlib.models", "pufferlib._C"),
)
def test_load_rejects_any_preloaded_puffer_module(
    monkeypatch,
    tmp_path: Path,
    module_name: str,
) -> None:
    _write_build_inputs(tmp_path)
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.subprocess, "run", _fake_successful_build(tmp_path))
    runner.build_environment(tmp_path, ENV_NAME)

    with _clean_puffer_modules():
        sys.modules[module_name] = ModuleType(module_name)
        with pytest.raises(RuntimeError, match="already loaded"):
            runner.load_puffer(tmp_path, ENV_NAME)


def test_load_rejects_extension_imported_from_wrong_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_python_puffer_package(tmp_path, "__file__ = '/tmp/wrong/_C.py'\n")
    monkeypatch.setattr(
        runner,
        "current_python_abi",
        lambda: {"ext_suffix": ".py", "cache_tag": "test-cache"},
    )
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.subprocess, "run", _fake_successful_build(tmp_path, None))
    runner.build_environment(tmp_path, ENV_NAME)

    with _clean_puffer_modules():
        with pytest.raises(RuntimeError, match="wrong path"):
            runner.load_puffer(tmp_path, ENV_NAME)


def test_load_rejects_compiled_environment_name(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_python_puffer_package(tmp_path, "env_name = 'another_env'\n")
    monkeypatch.setattr(
        runner,
        "current_python_abi",
        lambda: {"ext_suffix": ".py", "cache_tag": "test-cache"},
    )
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.subprocess, "run", _fake_successful_build(tmp_path, None))
    runner.build_environment(tmp_path, ENV_NAME)

    with _clean_puffer_modules():
        with pytest.raises(RuntimeError, match="another_env"):
            runner.load_puffer(tmp_path, ENV_NAME)


def _write_python_puffer_package(root: Path, extension_suffix: str) -> None:
    _write_build_inputs(root)
    package = root / "pufferlib"
    (package / "__init__.py").write_text("")
    (package / "models.py").write_text("")
    (package / "pufferl.py").write_text(
        "def load_config(name):\n"
        "    return {'env': {'env_name': name}, 'vec': {}}\n"
    )
    (package / "torch_pufferl.py").write_text("from . import _C\n")
    (package / "_C.py").write_text(
        f"env_name = {ENV_NAME!r}\ngpu = 0\n{extension_suffix}"
    )
