from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from time import time


class CrazyflieConsoleCapture:
    def __init__(self, crazyflie, output: str | Path | None) -> None:
        self.crazyflie = crazyflie
        self.output = None if output is None else Path(output)
        self._handle = None
        self._lock = Lock()

    def start(self) -> None:
        if self.output is None or self._handle is not None:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.output.open("w")
        self.crazyflie.console.receivedChar.add_callback(self._on_text)

    def close(self) -> None:
        if self._handle is None:
            return
        self.crazyflie.console.receivedChar.remove_callback(self._on_text)
        with self._lock:
            self._handle.close()
            self._handle = None

    def _on_text(self, text: str) -> None:
        with self._lock:
            if self._handle is None:
                return
            self._handle.write(json.dumps({"host_time_s": time(), "text": text}) + "\n")
            self._handle.flush()
