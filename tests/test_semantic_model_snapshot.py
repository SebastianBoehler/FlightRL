from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from flightrl.semantic import clip_verifier, grounding_dino
from flightrl.semantic.clip_verifier import ClipCropVerifier, ClipVerifierConfig
from flightrl.semantic.grounding_dino import (
    GroundingDinoConfig,
    GroundingDinoGrounder,
)
from flightrl.semantic.model_artifact import (
    verified_local_huggingface_snapshot,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_verified_snapshot_copies_only_manifest_and_survives_source_mutation(
    tmp_path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    approved = {
        "config.json": b'{"model_type":"approved"}',
        "pytorch_model.bin": b"approved-weights",
    }
    for filename, data in approved.items():
        (source / filename).write_bytes(data)
    (source / "model.safetensors").write_bytes(b"unapproved-alternate")
    calls: list[dict] = []

    def resolve(**kwargs):
        calls.append(kwargs)
        return source / kwargs["filename"]

    manifest = tuple((name, _sha(data)) for name, data in approved.items())
    with verified_local_huggingface_snapshot(
        model_id="owner/model",
        revision="a" * 40,
        manifest=manifest,
        runtime_versions=(),
        resolver=resolve,
    ) as snapshot:
        (source / "config.json").write_bytes(b"changed-after-copy")

        assert (snapshot / "config.json").read_bytes() == approved["config.json"]
        assert (snapshot / "pytorch_model.bin").read_bytes() == approved[
            "pytorch_model.bin"
        ]
        assert not (snapshot / "model.safetensors").exists()
        snapshot_path = snapshot

    assert not snapshot_path.exists()
    assert calls == [
        {
            "repo_id": "owner/model",
            "revision": "a" * 40,
            "filename": filename,
            "local_files_only": True,
        }
        for filename in ("config.json", "pytorch_model.bin")
    ]


def test_verified_snapshot_rejects_any_manifest_digest_mismatch(tmp_path) -> None:
    source = tmp_path / "config.json"
    source.write_bytes(b"changed")

    with pytest.raises(ValueError, match="SHA-256"):
        with verified_local_huggingface_snapshot(
            model_id="owner/model",
            revision="b" * 40,
            manifest=(("config.json", "0" * 64),),
            runtime_versions=(),
            resolver=lambda **_: source,
        ):
            pytest.fail("mismatched snapshot must not be yielded")


class _FakeModel:
    def __init__(self) -> None:
        self.device = None
        self.evaluating = False

    def to(self, device):
        self.device = device
        return self

    def eval(self) -> None:
        self.evaluating = True


class _FakeLoader:
    calls: list[tuple[tuple, dict]] = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.calls.append((args, kwargs))
        return _FakeModel()


@contextmanager
def _snapshot_at(path: Path, **_):
    yield path


def _fake_ml_modules(monkeypatch) -> None:
    _FakeLoader.calls = []
    fake_transformers = SimpleNamespace(
        AutoModelForZeroShotObjectDetection=_FakeLoader,
        AutoProcessor=_FakeLoader,
        CLIPModel=_FakeLoader,
    )
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False),
        )
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


@pytest.mark.parametrize(
    ("module", "config", "constructor", "expected_safetensors"),
    (
        (
            grounding_dino,
            GroundingDinoConfig(
                revision="c" * 40,
                artifact_manifest=(("model.safetensors", "1" * 64),),
                weights_format="safetensors",
            ),
            GroundingDinoGrounder,
            True,
        ),
        (
            clip_verifier,
            ClipVerifierConfig(
                revision="d" * 40,
                artifact_manifest=(("pytorch_model.bin", "2" * 64),),
                weights_format="pytorch",
                device="cpu",
            ),
            ClipCropVerifier,
            False,
        ),
    ),
)
def test_detector_loaders_consume_private_snapshot_with_explicit_format(
    tmp_path,
    monkeypatch,
    module,
    config,
    constructor,
    expected_safetensors,
) -> None:
    _fake_ml_modules(monkeypatch)
    snapshot = tmp_path / "private-snapshot"
    monkeypatch.setattr(
        module,
        "huggingface_model_source",
        lambda **kwargs: _snapshot_at(snapshot, **kwargs),
    )

    constructor(config)

    processor_call, model_call = _FakeLoader.calls
    assert processor_call == (
        (snapshot,),
        {
            "local_files_only": True,
            "trust_remote_code": False,
        },
    )
    assert model_call == (
        (snapshot,),
        {
            "local_files_only": True,
            "trust_remote_code": False,
            "use_safetensors": expected_safetensors,
            "weights_only": True,
        },
    )


@pytest.mark.parametrize(
    ("config", "constructor", "model_id"),
    (
        (
            GroundingDinoConfig(device="cpu"),
            GroundingDinoGrounder,
            "IDEA-Research/grounding-dino-tiny",
        ),
        (
            ClipVerifierConfig(device="cpu"),
            ClipCropVerifier,
            "openai/clip-vit-base-patch32",
        ),
    ),
)
def test_generic_detector_loading_preserves_automatic_weight_selection(
    monkeypatch,
    config,
    constructor,
    model_id,
) -> None:
    _fake_ml_modules(monkeypatch)

    constructor(config)

    processor_call, model_call = _FakeLoader.calls
    assert processor_call[0] == (model_id,)
    assert model_call[0] == (model_id,)
    assert "use_safetensors" not in model_call[1]


def test_pinned_detector_rejects_weight_format_manifest_mismatch() -> None:
    with pytest.raises(ValueError, match="weight artifact"):
        GroundingDinoConfig(
            revision="e" * 40,
            artifact_manifest=(("pytorch_model.bin", "3" * 64),),
            weights_format="safetensors",
        )
