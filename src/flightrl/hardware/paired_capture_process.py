from __future__ import annotations

from dataclasses import dataclass
import csv
from math import isfinite
import os
from pathlib import Path
import signal
import subprocess
from time import monotonic, sleep
from typing import Callable, IO, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    pid: int
    returncode: int


@dataclass(frozen=True, slots=True)
class BoundedProcessOutcome:
    pid: int
    returncode: int
    timed_out: bool
    elapsed_s: float

    @property
    def succeeded(self) -> bool:
        return not self.timed_out and self.returncode == 0


@dataclass(frozen=True, slots=True)
class PairedCaptureProcessOutcome:
    camera: ProcessOutcome
    telemetry: ProcessOutcome
    timed_out: bool
    elapsed_s: float

    @property
    def succeeded(self) -> bool:
        return (
            not self.timed_out
            and self.camera.returncode == 0
            and self.telemetry.returncode == 0
        )


def run_bounded_process(
    *,
    command: Sequence[str],
    timeout_s: float,
    cleanup_timeout_s: float,
    output: IO[str] | int | None = None,
) -> BoundedProcessOutcome:
    argv = _command("process", command)
    timeout = _duration("process timeout", timeout_s)
    cleanup_timeout = _duration("cleanup timeout", cleanup_timeout_s)
    started = monotonic()
    process = _start(argv, output)
    timed_out = False
    try:
        deadline = started + timeout
        while process.poll() is None:
            if monotonic() >= deadline:
                timed_out = True
                break
            sleep(min(0.01, max(0.0, deadline - monotonic())))
    finally:
        _bounded_cleanup([process], cleanup_timeout)
    returncode = process.poll()
    if returncode is None:
        raise RuntimeError("subprocess survived bounded cleanup")
    return BoundedProcessOutcome(
        pid=process.pid,
        returncode=returncode,
        timed_out=timed_out,
        elapsed_s=monotonic() - started,
    )


def run_bounded_capture_processes(
    *,
    camera_command: Sequence[str],
    telemetry_command: Sequence[str],
    telemetry_ready_path: str | Path,
    telemetry_required_columns: Sequence[str],
    telemetry_minimum_values: Mapping[str, float],
    telemetry_ready_timeout_s: float,
    timeout_s: float,
    cleanup_timeout_s: float,
    before_camera: Callable[[], None] | None = None,
    camera_output: IO[str] | int | None = None,
    telemetry_output: IO[str] | int | None = None,
) -> PairedCaptureProcessOutcome:
    ready_path = Path(telemetry_ready_path)
    required_columns = _columns(telemetry_required_columns)
    minimum_values = _minimum_values(telemetry_minimum_values, required_columns)
    ready_timeout = _duration("telemetry ready timeout", telemetry_ready_timeout_s)
    timeout = _duration("capture timeout", timeout_s)
    cleanup_timeout = _duration("cleanup timeout", cleanup_timeout_s)
    camera_argv = _command("camera", camera_command)
    telemetry_argv = _command("telemetry", telemetry_command)
    started = monotonic()
    telemetry = _start(telemetry_argv, telemetry_output)
    camera: subprocess.Popen[str] | None = None
    timed_out = False
    try:
        _wait_for_telemetry_ready(
            ready_path,
            required_columns,
            minimum_values,
            ready_timeout,
            telemetry,
        )
        if before_camera is not None:
            before_camera()
        camera = _start(camera_argv, camera_output)
        deadline = started + timeout
        while camera.poll() is None or telemetry.poll() is None:
            if camera.returncode not in (None, 0) or telemetry.returncode not in (None, 0):
                break
            if monotonic() >= deadline:
                timed_out = True
                break
            sleep(min(0.01, max(0.0, deadline - monotonic())))
    finally:
        processes = [telemetry] if camera is None else [camera, telemetry]
        _bounded_cleanup(processes, cleanup_timeout)
    if camera is None:
        raise RuntimeError("camera process did not start")
    camera_code = camera.poll()
    telemetry_code = telemetry.poll()
    if camera_code is None or telemetry_code is None:
        raise RuntimeError("capture subprocess survived bounded cleanup")
    return PairedCaptureProcessOutcome(
        camera=ProcessOutcome(camera.pid, camera_code),
        telemetry=ProcessOutcome(telemetry.pid, telemetry_code),
        timed_out=timed_out,
        elapsed_s=monotonic() - started,
    )


def _start(command: tuple[str, ...], output: IO[str] | int | None) -> subprocess.Popen[str]:
    target = subprocess.DEVNULL if output is None else output
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=target,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _wait_for_telemetry_ready(
    path: Path,
    required_columns: tuple[str, ...],
    minimum_values: dict[str, float],
    timeout_s: float,
    process: subprocess.Popen[str],
) -> None:
    deadline = monotonic() + timeout_s
    wrong_header_seen = False
    while True:
        if process.poll() is not None:
            raise RuntimeError("telemetry process exited before camera start")
        ready, wrong_header = _csv_readiness(path, required_columns, minimum_values)
        wrong_header_seen = wrong_header_seen or wrong_header
        if ready:
            return
        if monotonic() >= deadline:
            if wrong_header_seen:
                raise TimeoutError(
                    "telemetry did not produce the exact telemetry header before startup timeout"
                )
            raise TimeoutError(
                "telemetry did not produce its first data row before startup timeout"
            )
        sleep(min(0.01, max(0.0, deadline - monotonic())))


def _csv_readiness(
    path: Path,
    required_columns: tuple[str, ...],
    minimum_values: dict[str, float],
) -> tuple[bool, bool]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return False, False
    if not lines:
        return False, False
    header = tuple(next(csv.reader((lines[0],))))
    if header != required_columns:
        return False, True
    if len(lines) < 2:
        return False, False
    row = tuple(next(csv.reader((lines[1],))))
    if len(row) != len(required_columns):
        raise RuntimeError("telemetry first data row does not match the exact header")
    values = dict(zip(required_columns, row, strict=True))
    for name, minimum in minimum_values.items():
        try:
            value = float(values[name])
        except ValueError as exc:
            raise RuntimeError(f"telemetry first-row {name} is not numeric") from exc
        if not isfinite(value) or value < minimum:
            raise RuntimeError(
                f"telemetry first-row {name} must be finite and at least {minimum:.2f}"
            )
    return True, False


def _bounded_cleanup(processes: list[subprocess.Popen[str]], timeout_s: float) -> None:
    alive = [process for process in processes if process.poll() is None]
    for process in alive:
        os.killpg(process.pid, signal.SIGTERM)
    deadline = monotonic() + timeout_s
    while alive and monotonic() < deadline:
        alive = [process for process in alive if process.poll() is None]
        if alive:
            sleep(min(0.01, max(0.0, deadline - monotonic())))
    for process in alive:
        os.killpg(process.pid, signal.SIGKILL)
    for process in alive:
        process.wait(timeout=timeout_s)
    for process in processes:
        process.poll()


def _command(label: str, command: Sequence[str]) -> tuple[str, ...]:
    values = tuple(command)
    if not values or not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"{label} command must contain non-empty strings")
    return values


def _columns(columns: Sequence[str]) -> tuple[str, ...]:
    values = tuple(columns)
    if not values or len(set(values)) != len(values) or not all(
        isinstance(value, str) and value for value in values
    ):
        raise ValueError("telemetry required columns must be unique non-empty strings")
    return values


def _minimum_values(
    values: Mapping[str, float],
    required_columns: tuple[str, ...],
) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError("telemetry minimum values must be a mapping")
    result: dict[str, float] = {}
    for name, value in values.items():
        if name not in required_columns or isinstance(value, bool):
            raise ValueError("telemetry minimum values must name required numeric columns")
        minimum = float(value)
        if not isfinite(minimum):
            raise ValueError("telemetry minimum values must be finite")
        result[name] = minimum
    return result


def _duration(label: str, value: float, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite and positive")
    result = float(value)
    if not isfinite(result) or result < 0.0 or (not allow_zero and result == 0.0):
        raise ValueError(f"{label} must be finite and positive")
    return result
