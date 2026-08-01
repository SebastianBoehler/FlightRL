from __future__ import annotations

from threading import Condition, Event, Thread
from time import monotonic, time
from typing import Mapping

import numpy as np

from flightrl.hardware.aideck_stream import (
    AIDECK_GRAY4_FORMAT,
    AiDeckUdpStream,
)
from flightrl.puffer4_vision_runtime import VisualPufferShadow


class VisualPolicyWorkerError(RuntimeError):
    pass


class VisualPolicyWorker:
    """Continuously runs recurrent inference at the AI Deck frame rate."""

    def __init__(
        self,
        checkpoint,
        *,
        stream: AiDeckUdpStream,
        initial_intent: np.ndarray,
    ) -> None:
        self.shadow = VisualPufferShadow(checkpoint)
        self.stream = stream
        self._intent = np.asarray(initial_intent, dtype=np.float32).copy()
        self._condition = Condition()
        self._stop = Event()
        self._thread: Thread | None = None
        self._latest: dict[str, float | int | bool] | None = None
        self._error: Exception | None = None
        self._frame_count = 0
        self._reset_requested = False
        self._generation = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("visual policy worker already started")
        self._thread = Thread(target=self._run, name="visual-policy", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.stream.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def set_intent(self, intent: np.ndarray) -> None:
        values = np.asarray(intent, dtype=np.float32)
        if values.shape != (6,):
            raise ValueError("visual policy intent must have shape (6,)")
        with self._condition:
            self._intent = values.copy()

    def reset(self, intent: np.ndarray) -> None:
        self.set_intent(intent)
        with self._condition:
            self._generation += 1
            self._reset_requested = True
            self._frame_count = 0
            self._latest = None

    def wait_for_frames(
        self,
        minimum_frames: int,
        *,
        timeout_s: float,
    ) -> Mapping[str, float | int | bool]:
        deadline = monotonic() + timeout_s
        with self._condition:
            while self._frame_count < minimum_frames and self._error is None:
                remaining = deadline - monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"visual policy produced {self._frame_count}/{minimum_frames} frames"
                    )
                self._condition.wait(remaining)
            self._raise_error()
            if self._latest is None:
                raise RuntimeError("visual policy has no completed frame")
            return dict(self._latest)

    def snapshot(self) -> Mapping[str, float | int | bool]:
        with self._condition:
            self._raise_error()
            if self._latest is None:
                raise RuntimeError("visual policy has no completed frame")
            return dict(self._latest)

    @property
    def frame_count(self) -> int:
        with self._condition:
            return self._frame_count

    def _run(self) -> None:
        try:
            with self.stream:
                while not self._stop.is_set():
                    frame = self.stream.read_frame()
                    if (
                        frame.width != 64
                        or frame.height != 48
                        or frame.format != AIDECK_GRAY4_FORMAT
                    ):
                        raise RuntimeError(
                            "active visual control requires the frame-safe "
                            "64x48 gray4 camera profile"
                        )
                    with self._condition:
                        if self._reset_requested:
                            self.shadow.reset()
                            self._reset_requested = False
                        generation = self._generation
                        intent = self._intent.copy()
                    prediction = self.shadow.step(frame.pixels, intent)
                    snapshot = {
                        **prediction,
                        "frame_index": frame.index,
                        "frame_host_time_s": frame.host_time_s,
                        "frame_mean": float(frame.pixels.mean()),
                        "frame_width": frame.width,
                        "frame_height": frame.height,
                        "frame_format": frame.format,
                        "dropped_frames": self.stream.dropped_frames,
                        "worker_host_time_s": time(),
                    }
                    with self._condition:
                        if generation != self._generation:
                            continue
                        self._latest = snapshot
                        self._frame_count += 1
                        self._condition.notify_all()
        except Exception as exc:
            if not self._stop.is_set():
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()

    def _raise_error(self) -> None:
        if self._error is not None:
            raise VisualPolicyWorkerError(
                f"visual policy worker failed: {self._error}"
            ) from self._error
