from __future__ import annotations

import json
from types import SimpleNamespace

from flightrl.hardware.console_capture import CrazyflieConsoleCapture


class _Callbacks:
    def __init__(self) -> None:
        self.callbacks = []

    def add_callback(self, callback) -> None:
        self.callbacks.append(callback)

    def remove_callback(self, callback) -> None:
        self.callbacks.remove(callback)


def test_console_capture_writes_timestamped_jsonl(tmp_path) -> None:
    callbacks = _Callbacks()
    crazyflie = SimpleNamespace(console=SimpleNamespace(receivedChar=callbacks))
    output = tmp_path / "console.jsonl"
    capture = CrazyflieConsoleCapture(crazyflie, output)

    capture.start()
    callbacks.callbacks[0]("CPX: GAP8: capture=65ms\n")
    capture.close()

    row = json.loads(output.read_text())
    assert row["text"] == "CPX: GAP8: capture=65ms\n"
    assert row["host_time_s"] > 0
    assert callbacks.callbacks == []
