from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.sim2real.live_profile import build_live_sim_profile, write_report


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 6-DoF simulator sensor profile from rich Crazyflie logs.")
    parser.add_argument("--flight-log", action="append", type=Path, default=[], help="Rich flight CSV. Repeatable.")
    parser.add_argument("--stationary-log", action="append", type=Path, default=[], help="Stationary rich CSV. Repeatable.")
    parser.add_argument("--latency-report", type=Path, default=None)
    parser.add_argument("--name", default="rich_live_20260619")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/replay/rich_live_sim_profile_20260619.json")
    args = parser.parse_args()

    flight_logs = args.flight_log or default_flight_logs()
    stationary_logs = args.stationary_log or default_stationary_logs()
    report = build_live_sim_profile(
        flight_logs=flight_logs,
        stationary_logs=stationary_logs,
        latency_report=args.latency_report,
        name=args.name,
    )
    write_report(report, args.output)
    profile = report["sensor_profile"]
    print(f"profile={args.output}")
    print(
        "range_noise_std_m={:.4f} dropout={:.4f} action_lag_s={:.4f}".format(
            profile["range_noise_std_m"],
            profile["range_dropout_prob"],
            profile["action_lag_s"],
        )
    )


def default_flight_logs() -> list[Path]:
    logs = ROOT / "artifacts/crazyflie_logs"
    return sorted(path for path in logs.glob("rich_*.csv") if "stationary" not in path.name and "battery_check" not in path.name)


def default_stationary_logs() -> list[Path]:
    logs = ROOT / "artifacts/crazyflie_logs"
    return sorted(path for path in logs.glob("rich_stationary*.csv")) + sorted(path for path in logs.glob("stationary_post_run27*.csv"))


if __name__ == "__main__":
    main()
