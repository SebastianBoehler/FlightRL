"""Measure renderer-independent serial environment stepping; excludes image capture."""

import argparse
import json
import platform
import resource
import time
from pathlib import Path
import numpy as np
from flightrl.robotics.environment import RobotEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for count in (1, 4, 8):
        reset_start = time.perf_counter()
        environments = [RobotEnvironment(120 + i, industry=True) for i in range(count)]
        reset_ms = (time.perf_counter() - reset_start) * 1000 / count
        for env in environments:
            for _ in range(10):
                env.step()
        timings = []
        start = time.perf_counter()
        for _ in range(150):
            tick = time.perf_counter()
            for env in environments:
                env.step()
            timings.append(time.perf_counter() - tick)
        elapsed = time.perf_counter() - start
        results.append(
            dict(
                environments=count,
                reset_ms_per_environment=reset_ms,
                wall_s=elapsed,
                control_steps_per_s=150 * count / elapsed,
                physics_substeps_per_s=1500 * count / elapsed,
                batch_p95_ms=float(np.quantile(timings, 0.95) * 1000),
                aggregate_simulation_realtime_factor=3 * count / elapsed,
            )
        )
    report = dict(
        machine=platform.machine(),
        python=platform.python_version(),
        peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if platform.system() == "Darwin" else 1024),
        results=results,
        scope="Serial CPU MuJoCo physics, noisy proprioception and state serialization. No camera, rendering or visual policy; not end-to-end training throughput.",
    )
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
