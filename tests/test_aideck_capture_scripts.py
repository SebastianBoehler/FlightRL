from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
USB_CAPTURE_CONFIG = (
    "configs/hardware/crazyflie_2_1_brushless_aideck_flow2_usb_capture.toml"
)
FLOW_PREFLIGHT_CONFIG = (
    "configs/hardware/crazyflie_2_1_brushless_aideck_flow2_usb_flow_preflight.toml"
)


def test_paired_capture_dry_run_uses_bounded_usb_telemetry_profile(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/capture_aideck_with_telemetry.py",
            "--config",
            USB_CAPTURE_CONFIG,
            "--run-dir",
            str(tmp_path / "paired"),
            "--duration-s",
            "23",
            "--frames",
            "1200",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usb://0" in result.stdout
    assert '"overall_timeout_s": 38.0' in result.stdout
    assert '"controls_drone": false' in result.stdout
    assert not (tmp_path / "paired").exists()


def test_paired_capture_rejects_flow_preflight_profile_before_live_run(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/capture_aideck_with_telemetry.py",
            "--config",
            FLOW_PREFLIGHT_CONFIG,
            "--run-dir",
            str(tmp_path / "paired"),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "exact five-variable stationary profile" in result.stderr
    assert not (tmp_path / "paired").exists()


def test_paired_capture_rejects_telemetry_window_shorter_than_camera_envelope(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/capture_aideck_with_telemetry.py",
            "--config",
            USB_CAPTURE_CONFIG,
            "--run-dir",
            str(tmp_path / "paired"),
            "--duration-s",
            "20",
            "--frames",
            "1200",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "exactly 23" in result.stderr
    assert not (tmp_path / "paired").exists()


def test_paired_capture_rejects_nonstandard_long_telemetry_window(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/capture_aideck_with_telemetry.py",
            "--config",
            USB_CAPTURE_CONFIG,
            "--run-dir",
            str(tmp_path / "paired"),
            "--duration-s",
            "600",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "exactly 23" in result.stderr
    assert not (tmp_path / "paired").exists()


def test_paired_capture_rejects_nonstandard_frame_count_before_live_run(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/capture_aideck_with_telemetry.py",
            "--config",
            USB_CAPTURE_CONFIG,
            "--run-dir",
            str(tmp_path / "paired"),
            "--frames",
            "100",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "exactly 1200 frames" in result.stderr
    assert not (tmp_path / "paired").exists()


def test_paired_capture_rejects_wrong_deck_profile_before_live_run(
    tmp_path: Path,
) -> None:
    source = (ROOT / USB_CAPTURE_CONFIG).read_text()
    config = tmp_path / "wrong-decks.toml"
    config.write_text(source.replace("expect_ai_deck = true", "expect_ai_deck = false"))
    result = subprocess.run(
        [
            sys.executable,
            "scripts/capture_aideck_with_telemetry.py",
            "--config",
            str(config),
            "--run-dir",
            str(tmp_path / "paired"),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "exact AI Deck, Flow Deck, and Z-ranger stack" in result.stderr
    assert not (tmp_path / "paired").exists()


def test_paired_capture_rejects_nonzero_usb_device_before_live_run(
    tmp_path: Path,
) -> None:
    source = (ROOT / USB_CAPTURE_CONFIG).read_text()
    config = tmp_path / "wrong-usb.toml"
    config.write_text(source.replace('uri = "usb://0"', 'uri = "usb://7"'))

    result = subprocess.run(
        [
            sys.executable,
            "scripts/capture_aideck_with_telemetry.py",
            "--config",
            str(config),
            "--run-dir",
            str(tmp_path / "paired"),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "exact usb://0" in result.stderr
    assert not (tmp_path / "paired").exists()


def test_paired_capture_rejects_nonstandard_camera_endpoint_before_live_run(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/capture_aideck_with_telemetry.py",
            "--config",
            USB_CAPTURE_CONFIG,
            "--run-dir",
            str(tmp_path / "paired"),
            "--host",
            "192.168.4.9",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "exact AI Deck UDP endpoint" in result.stderr
    assert not (tmp_path / "paired").exists()


def test_paired_capture_rejects_missing_flow_preflight_report_before_live_run(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/capture_aideck_with_telemetry.py",
            "--config",
            USB_CAPTURE_CONFIG,
            "--run-dir",
            str(tmp_path / "paired"),
            "--flow-preflight-report",
            str(tmp_path / "missing-preflight.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "fresh passing props-off Flow preflight" in result.stderr
    assert not (tmp_path / "paired").exists()


def test_legacy_radio_ground_gate_fails_before_starting_capture() -> None:
    result = subprocess.run(
        ["bash", "scripts/aideck_udp_ground_gate.sh", "radio"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "capture_aideck_with_telemetry.py" in result.stderr
    assert "radio mode is disabled" in result.stderr
