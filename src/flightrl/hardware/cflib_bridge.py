from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Callable

from .config import CrazyflieHardwareConfig
from .errors import HardwareDependencyError


@dataclass(slots=True)
class CflibModules:
    crtp: ModuleType
    crazyflie_cls: type
    sync_crazyflie_cls: type
    motion_commander_cls: type
    log_config_cls: type
    sync_logger_cls: type


def require_cflib(import_module: Callable[[str], ModuleType] = import_module) -> CflibModules:
    try:
        crtp = import_module("cflib.crtp")
        crazyflie_mod = import_module("cflib.crazyflie")
        sync_mod = import_module("cflib.crazyflie.syncCrazyflie")
        motion_mod = import_module("cflib.positioning.motion_commander")
        log_mod = import_module("cflib.crazyflie.log")
        logger_mod = import_module("cflib.crazyflie.syncLogger")
    except ModuleNotFoundError as exc:
        raise HardwareDependencyError(
            'Crazyflie hardware commands require cflib. Install it with: python -m pip install -e ".[hardware]"'
        ) from exc

    return CflibModules(
        crtp=crtp,
        crazyflie_cls=crazyflie_mod.Crazyflie,
        sync_crazyflie_cls=sync_mod.SyncCrazyflie,
        motion_commander_cls=motion_mod.MotionCommander,
        log_config_cls=log_mod.LogConfig,
        sync_logger_cls=logger_mod.SyncLogger,
    )


def init_drivers(modules: CflibModules | None = None) -> None:
    modules = modules or require_cflib()
    modules.crtp.init_drivers()


def scan_interfaces(modules: CflibModules | None = None) -> list[str]:
    modules = modules or require_cflib()
    init_drivers(modules)
    interfaces = modules.crtp.scan_interfaces()
    uris: list[str] = []
    for item in interfaces:
        uris.append(str(item[0] if isinstance(item, (tuple, list)) else item))
    return uris


def sync_crazyflie_context(config: CrazyflieHardwareConfig, modules: CflibModules | None = None):
    modules = modules or require_cflib()
    init_drivers(modules)
    crazyflie = modules.crazyflie_cls(rw_cache=config.radio.cache_dir)
    return modules.sync_crazyflie_cls(config.radio.uri, cf=crazyflie)
