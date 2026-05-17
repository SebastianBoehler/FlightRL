from __future__ import annotations

from flightrl.hardware.telemetry import TelemetryCsvWriter, TelemetrySample


def test_telemetry_csv_writes_replay_friendly_rows(tmp_path) -> None:
    path = tmp_path / "flight.csv"
    writer = TelemetryCsvWriter(path, variables=("stabilizer.roll", "pm.vbat"))
    writer.write_sample(
        TelemetrySample(
            host_time_s=1.25,
            crazyflie_time_ms=50,
            values={"stabilizer.roll": 2.0, "pm.vbat": 3.85},
        )
    )
    writer.close()

    assert path.read_text().splitlines() == [
        "host_time_s,crazyflie_time_ms,stabilizer.roll,pm.vbat",
        "1.250000,50,2.0,3.85",
    ]
