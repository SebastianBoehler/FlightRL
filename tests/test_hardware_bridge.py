from __future__ import annotations

import pytest

from flightrl.hardware.cflib_bridge import require_cflib
from flightrl.hardware.errors import HardwareDependencyError


def test_missing_cflib_has_actionable_error() -> None:
    with pytest.raises(HardwareDependencyError, match=r"\.\[hardware\]"):
        require_cflib(import_module=lambda _: (_ for _ in ()).throw(ModuleNotFoundError("cflib")))
