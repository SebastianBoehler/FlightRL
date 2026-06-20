from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.sim2real.live_system_id import build_live_system_id_report, write_report


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build command-response system ID from rich Crazyflie live logs.")
    parser.add_argument("--flight-log", action="append", type=Path, default=[], help="Rich flight CSV. Repeatable.")
    parser.add_argument("--base-profile", type=Path, default=None, help="Existing rich live sim profile JSON to refine.")
    parser.add_argument("--name", default="live_system_id")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/replay/live_system_id.json")
    args = parser.parse_args()

    flight_logs = args.flight_log or default_flight_logs()
    report = build_live_system_id_report(flight_logs=flight_logs, base_profile=args.base_profile, name=args.name)
    write_report(report, args.output)
    response = report["response"]
    print(f"system_id={args.output}")
    print(
        "lag_median_s={:.4f} tau_median_s={:.4f} gain_median={:.4f}".format(
            response["lag_s"]["median"] or 0.0,
            response["tau_s"]["median"] or 0.0,
            response["gain"]["median"] or 0.0,
        )
    )


def default_flight_logs() -> list[Path]:
    logs = ROOT / "artifacts/crazyflie_logs"
    return sorted(path for path in logs.glob("rich_reactive_escape_brake*.csv"))


if __name__ == "__main__":
    main()
