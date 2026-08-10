from __future__ import annotations

import hashlib
import json
import struct

import numpy as np

from .student_sequence import (
    COVERAGE_SEQUENCE_ARRAYS,
    CoverageSequenceDataset,
    require_coverage_sequence_dataset,
)


COVERAGE_SEQUENCE_DIGEST_SCHEMA = "flightrl.coverage.sequence_sha256.v1"


def coverage_sequence_sha256(dataset: CoverageSequenceDataset) -> str:
    """Bind canonical metadata and every exact named array byte."""
    require_coverage_sequence_dataset(dataset)
    digest = hashlib.sha256()
    digest.update(COVERAGE_SEQUENCE_DIGEST_SCHEMA.encode("ascii") + b"\0")
    metadata = json.dumps(
        dataset.metadata,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")
    _section(digest, metadata)
    for name in COVERAGE_SEQUENCE_ARRAYS:
        array = np.ascontiguousarray(getattr(dataset, name))
        header = json.dumps(
            (name, array.dtype.str, list(array.shape)),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        _section(digest, header)
        _section(digest, array.tobytes(order="C"))
    return digest.hexdigest()


def _section(digest, value: bytes) -> None:
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)
