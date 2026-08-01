from __future__ import annotations

import numpy as np
import pytest

from flightrl.puffer4_door_evidence import (
    detector_evidence,
    observable_teacher_action,
)
from flightrl.semantic.contract import GroundingDetection, NormalizedBox


def test_detector_evidence_uses_verified_confidence_and_normalized_geometry() -> None:
    detection = GroundingDetection(
        "interior door",
        0.8,
        NormalizedBox(0.25, 0.20, 0.75, 0.60),
        verification_confidence=0.9,
    )

    evidence = detector_evidence(detection, age_s=0.25, maximum_age_s=1.0)

    np.testing.assert_allclose(evidence, (0.9, 0.0, -0.2, np.sqrt(0.2), 0.25))


def test_detector_evidence_represents_fresh_miss_and_stale_absence() -> None:
    fresh = detector_evidence(None, age_s=0.0, maximum_age_s=1.0)
    absent = detector_evidence(None, age_s=None, maximum_age_s=1.0)

    np.testing.assert_allclose(fresh, (0.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(absent, (0.0, 0.0, 0.0, 0.0, 1.0))


def test_detector_evidence_rejects_invalid_age_contract() -> None:
    with pytest.raises(ValueError, match="maximum_age_s"):
        detector_evidence(None, age_s=0.0, maximum_age_s=0.0)


def test_observable_teacher_searches_centers_approaches_and_stops() -> None:
    search = observable_teacher_action(np.zeros(5), target_seen=False)
    turn = observable_teacher_action(
        np.asarray((0.9, 0.5, 0.0, 0.3, 0.0)),
        target_seen=True,
    )
    approach = observable_teacher_action(
        np.asarray((0.9, 0.0, 0.0, 0.3, 0.0)),
        target_seen=True,
    )
    stop = observable_teacher_action(
        np.asarray((0.9, 0.0, 0.0, 0.98, 0.0)),
        target_seen=True,
    )

    np.testing.assert_allclose(search, (0.0, 0.85))
    assert turn[0] == 0.0
    assert turn[1] < 0.0
    assert approach[0] > 0.6
    np.testing.assert_allclose(stop, (0.0, 0.0))
