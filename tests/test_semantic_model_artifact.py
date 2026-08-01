from __future__ import annotations

import pytest

from flightrl.semantic import model_artifact


def test_optional_snapshot_requires_revision_and_nonempty_manifest() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        model_artifact.validate_optional_huggingface_snapshot(
            model_id="owner/model",
            revision=None,
            manifest=(("config.json", "0" * 64),),
            runtime_versions=(),
        )

    with pytest.raises(ValueError, match="empty"):
        model_artifact.validate_optional_huggingface_snapshot(
            model_id="owner/model",
            revision="a" * 40,
            manifest=(),
            runtime_versions=(),
        )


def test_verified_snapshot_rejects_runtime_version_drift(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "config.json"
    artifact.write_bytes(b"{}")
    monkeypatch.setattr(
        model_artifact.metadata,
        "version",
        lambda package: "changed",
    )

    with pytest.raises(ValueError, match="version does not match"):
        with model_artifact.verified_local_huggingface_snapshot(
            model_id="owner/model",
            revision="b" * 40,
            manifest=(
                (
                    "config.json",
                    "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
                ),
            ),
            runtime_versions=(("transformers", "approved"),),
            resolver=lambda **_: artifact,
        ):
            pytest.fail("runtime drift must fail before snapshot creation")
