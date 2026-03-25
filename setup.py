from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from setuptools import Extension, setup


ROOT = Path(__file__).parent.resolve()
DEBUG = os.environ.get("DEBUG", "0") == "1"

common_args = ["-O0", "-g"] if DEBUG else ["-O3"]
if os.name != "nt" and not DEBUG:
    common_args.extend(["-ffast-math"])

extension = Extension(
    "flightrl._binding",
    sources=[
        "src/flightrl/native/binding.c",
        "src/flightrl/native/native_actions.c",
        "src/flightrl/native/native_env.c",
        "src/flightrl/native/native_logging.c",
        "src/flightrl/native/native_observation.c",
        "src/flightrl/native/native_reset.c",
        "src/flightrl/native/native_reward.c",
        "src/flightrl/native/native_tasks.c",
        "src/flightrl/native/native_termination.c",
        "src/flightrl/native/native_dynamics.c",
        "src/flightrl/native/native_wind.c",
    ],
    include_dirs=[
        str(ROOT / "src" / "flightrl" / "native"),
        np.get_include(),
    ],
    extra_compile_args=common_args,
)

setup(ext_modules=[extension])
