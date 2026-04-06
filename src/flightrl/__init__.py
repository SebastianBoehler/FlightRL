from .config import FlightConfig, load_config
from .factory import make_env

__all__ = [
    "DronePlanarEnv",
    "FlightConfig",
    "load_config",
    "make_env",
]


def __getattr__(name: str):
    if name == "DronePlanarEnv":
        from .env import DronePlanarEnv

        return DronePlanarEnv
    raise AttributeError(name)
