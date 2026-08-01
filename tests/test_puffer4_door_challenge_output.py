from __future__ import annotations

from pathlib import Path

import pytest

from flightrl.puffer4_door_challenge_evaluation import (
    resolve_canonical_output,
    resolve_challenge_output,
    validate_challenge_options,
)


def test_challenge_options_reject_missing_control_and_combined_intervention(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control.json"

    with pytest.raises(ValueError, match="requires --control-report"):
        validate_challenge_options(
            challenge="pixel-noise",
            control_report=None,
            output=None,
            live_yaw_cap_challenge=False,
        )
    with pytest.raises(ValueError, match="combined"):
        validate_challenge_options(
            challenge="pixel-noise",
            control_report=control,
            output=None,
            live_yaw_cap_challenge=True,
        )
    with pytest.raises(ValueError, match="requires --challenge"):
        validate_challenge_options(
            challenge=None,
            control_report=control,
            output=None,
            live_yaw_cap_challenge=False,
        )
    validate_challenge_options(
        challenge=None,
        control_report=None,
        output=tmp_path / "retry.json",
        live_yaw_cap_challenge=False,
    )


def test_challenge_output_is_distinct_and_never_overwritten(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "door.bin"
    control = checkpoint.with_suffix(".promotion-evaluation.json")
    lineage = tmp_path / "lineage.json"
    default = resolve_challenge_output(
        checkpoint,
        "pixel-noise",
        control_report=control,
        lineage_report=lineage,
        requested=None,
    )

    assert default == checkpoint.with_suffix(
        ".pixel-noise.challenge-evaluation.json"
    ).resolve()
    with pytest.raises(ValueError, match="canonical"):
        resolve_challenge_output(
            checkpoint,
            "pixel-noise",
            control_report=control,
            lineage_report=lineage,
            requested=control,
        )
    with pytest.raises(ValueError, match="lineage"):
        resolve_challenge_output(
            checkpoint,
            "pixel-noise",
            control_report=control,
            lineage_report=lineage,
            requested=lineage,
        )
    default.write_text("{}")
    with pytest.raises(FileExistsError, match="overwrite"):
        resolve_challenge_output(
            checkpoint,
            "pixel-noise",
            control_report=control,
            lineage_report=lineage,
            requested=default,
        )


def test_canonical_output_is_exclusive_and_cannot_alias_lineage(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "door.bin"
    lineage = tmp_path / "lineage.json"
    default = resolve_canonical_output(
        checkpoint,
        lineage_report=lineage,
        requested=None,
    )

    assert default == checkpoint.with_suffix(
        ".promotion-evaluation.json"
    ).resolve()
    with pytest.raises(ValueError, match="lineage"):
        resolve_canonical_output(
            checkpoint,
            lineage_report=lineage,
            requested=lineage,
        )
    default.write_text("{}")
    with pytest.raises(FileExistsError, match="overwrite"):
        resolve_canonical_output(
            checkpoint,
            lineage_report=lineage,
            requested=None,
        )
