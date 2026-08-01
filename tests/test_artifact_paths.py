from __future__ import annotations

import os

import pytest

from flightrl.artifact_paths import require_distinct_artifact_paths


def test_artifact_paths_resolve_distinct_outputs(tmp_path) -> None:
    resolved = require_distinct_artifact_paths(
        source=tmp_path / "source.bin",
        output=tmp_path / "nested/../output.bin",
    )

    assert resolved == {
        "source": (tmp_path / "source.bin").resolve(),
        "output": (tmp_path / "output.bin").resolve(),
    }


def test_artifact_paths_reject_resolved_alias(tmp_path) -> None:
    source = tmp_path / "source.bin"

    with pytest.raises(ValueError, match="source and output"):
        require_distinct_artifact_paths(
            source=source,
            output=tmp_path / "nested/../source.bin",
        )


def test_artifact_paths_reject_existing_hardlink_alias(tmp_path) -> None:
    source = tmp_path / "source.bin"
    alias = tmp_path / "alias.bin"
    source.write_bytes(b"bound evidence")
    os.link(source, alias)

    with pytest.raises(ValueError, match="source and output"):
        require_distinct_artifact_paths(source=source, output=alias)

    assert source.read_bytes() == b"bound evidence"
