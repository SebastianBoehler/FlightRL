from __future__ import annotations

import pytest

from flightrl.hardware.cflib_bridge import CflibModules, require_cflib, scan_interfaces
from flightrl.hardware.errors import HardwareDependencyError


def test_missing_cflib_has_actionable_error() -> None:
    with pytest.raises(HardwareDependencyError, match=r"\.\[hardware\]"):
        require_cflib(import_module=lambda _: (_ for _ in ()).throw(ModuleNotFoundError("cflib")))


class FakeCrtp:
    @staticmethod
    def init_drivers() -> None:
        pass

    @staticmethod
    def scan_interfaces():
        return [["radio://0/80/2M", ""], ("radio://0/90/2M/E7E7E7E7E7", "")]


def test_scan_interfaces_normalizes_cflib_pairs() -> None:
    modules = CflibModules(
        crtp=FakeCrtp,
        crazyflie_cls=object,
        sync_crazyflie_cls=object,
        motion_commander_cls=object,
        log_config_cls=object,
        sync_logger_cls=object,
    )

    assert scan_interfaces(modules) == ["radio://0/80/2M", "radio://0/90/2M/E7E7E7E7E7"]
