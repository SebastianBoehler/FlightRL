from __future__ import annotations

from flightrl import _binding


def test_native_core_exposes_versioned_c_contract() -> None:
    assert _binding.core_abi_version() == 1
