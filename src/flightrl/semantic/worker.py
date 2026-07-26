from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Protocol

from flightrl.hardware.aideck_stream import AiDeckFrame

from .contract import GroundingResult


class FrameStream(Protocol):
    def connect(self) -> None: ...
    def close(self) -> None: ...
    def read_frame(self) -> AiDeckFrame: ...


class FrameGrounder(Protocol):
    def detect(
        self,
        pixels,
        prompt: str,
        *,
        frame_index: int,
        frame_host_time_s: float,
    ) -> GroundingResult: ...


class AsyncGroundingPipeline:
    def __init__(self, stream: FrameStream, grounder: FrameGrounder, prompt: str) -> None:
        self.stream = stream
        self.grounder = grounder
        self.prompt = prompt
        self._frames: Queue[AiDeckFrame] = Queue(maxsize=1)
        self._stop = Event()
        self._first_frame = Event()
        self._first_result = Event()
        self._lock = Lock()
        self._latest_frame: AiDeckFrame | None = None
        self._latest_processed: tuple[AiDeckFrame, GroundingResult] | None = None
        self._error: BaseException | None = None
        self._threads: tuple[Thread, Thread] | None = None

    def start(self) -> None:
        if self._threads is not None:
            raise RuntimeError("grounding pipeline is already running")
        self.stream.connect()
        capture = Thread(target=self._capture_loop, name="aideck-capture", daemon=True)
        detect = Thread(target=self._detect_loop, name="semantic-grounder", daemon=True)
        self._threads = (capture, detect)
        capture.start()
        detect.start()

    def wait_for_frame(self, timeout_s: float) -> AiDeckFrame:
        if not self._first_frame.wait(timeout_s):
            self._raise_error()
            raise TimeoutError("timed out waiting for the first AI Deck frame")
        self._raise_error()
        with self._lock:
            assert self._latest_frame is not None
            return self._latest_frame

    def wait_for_result(self, timeout_s: float) -> tuple[AiDeckFrame, GroundingResult]:
        if not self._first_result.wait(timeout_s):
            self._raise_error()
            raise TimeoutError("timed out waiting for the first grounding result")
        latest = self.latest()
        assert latest is not None
        return latest

    def latest(self) -> tuple[AiDeckFrame, GroundingResult] | None:
        self._raise_error()
        with self._lock:
            return self._latest_processed

    def close(self) -> None:
        self._stop.set()
        self.stream.close()
        if self._threads is not None:
            for thread in self._threads:
                thread.join(timeout=2.0)
        self._threads = None

    def _capture_loop(self) -> None:
        try:
            while not self._stop.is_set():
                frame = self.stream.read_frame()
                with self._lock:
                    self._latest_frame = frame
                self._first_frame.set()
                try:
                    self._frames.put_nowait(frame)
                except Full:
                    try:
                        self._frames.get_nowait()
                    except Empty:
                        pass
                    self._frames.put_nowait(frame)
        except BaseException as exc:
            if not self._stop.is_set():
                self._set_error(exc)

    def _detect_loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    frame = self._frames.get(timeout=0.1)
                except Empty:
                    continue
                result = self.grounder.detect(
                    frame.pixels,
                    self.prompt,
                    frame_index=frame.index,
                    frame_host_time_s=frame.host_time_s,
                )
                with self._lock:
                    self._latest_processed = (frame, result)
                self._first_result.set()
        except BaseException as exc:
            if not self._stop.is_set():
                self._set_error(exc)

    def _set_error(self, exc: BaseException) -> None:
        with self._lock:
            self._error = exc
        self._stop.set()
        self._first_frame.set()
        self._first_result.set()

    def _raise_error(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"asynchronous grounding pipeline failed: {error}") from error
