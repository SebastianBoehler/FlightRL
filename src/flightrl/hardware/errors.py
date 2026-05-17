from __future__ import annotations


class HardwareError(RuntimeError):
    """Base class for hardware bring-up failures."""


class HardwareConfigError(HardwareError):
    """Raised when a Crazyflie hardware config is invalid."""


class HardwareDependencyError(HardwareError):
    """Raised when optional hardware dependencies are missing."""


class HardwareSafetyError(HardwareError):
    """Raised when a requested hardware action violates safety gates."""
