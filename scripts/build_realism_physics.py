"""Build the pinned Jolt bridge without changing the existing simulator extension."""

import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "src/flightrl/native/realism"
build = root / "build/realism"
subprocess.run(
    ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
    check=True,
)
subprocess.run(
    ["cmake", "--build", str(build), "--config", "Release", "--parallel", "4"],
    check=True,
)
print(f"Built native contacts in {build}")
