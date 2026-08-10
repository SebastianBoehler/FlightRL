from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from flightrl.hardware.errors import HardwareSafetyError


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/hardware/crazyflie_2_1_brushless_aideck_flow2_usb_flow_preflight.toml"
HEADER = (
    "host_time_s,crazyflie_time_ms,motion.motion,motion.deltaX,"
    "motion.deltaY,motion.squal,range.zrange"
)


def _flow_csv(*, healthy: bool) -> str:
    rows = [HEADER]
    for index in range(101):
        moving = 20 <= index < 40
        motion_status = 0xB0 if healthy else 0x30
        delta_x = 3 if healthy and moving else 0
        delta_y = -2 if healthy and moving else 0
        squal = 104 if healthy else 2
        rows.append(
            f"{100.0 + index * 0.05:.6f},{index * 50},{motion_status},"
            f"{delta_x},{delta_y},{squal},310"
        )
    return "\n".join(rows) + "\n"


def _writer_command(path: Path, payload: str) -> tuple[str, ...]:
    code = f"from pathlib import Path; Path({str(path)!r}).write_text({payload!r})"
    return (sys.executable, "-c", code)


def test_flow_preflight_sequence_validates_real_csv_after_both_children_succeed(
    tmp_path: Path,
) -> None:
    process = importlib.import_module("flightrl.hardware.flow_preflight_process")
    csv_path = tmp_path / "flow.csv"
    log_path = tmp_path / "telemetry.log"
    log_path.write_text("")

    outcome = process.run_flow_preflight_processes(
        deck_check_command=(sys.executable, "-c", "pass"),
        telemetry_command=_writer_command(csv_path, _flow_csv(healthy=True)),
        telemetry_path=csv_path,
        telemetry_log_path=log_path,
        deck_check_timeout_s=1.0,
        telemetry_timeout_s=1.0,
        cleanup_timeout_s=0.1,
        before_telemetry=lambda: None,
    )

    assert outcome.succeeded is True
    assert outcome.deck_check.returncode == 0
    assert outcome.telemetry is not None
    assert outcome.telemetry.returncode == 0
    assert outcome.validation is not None
    assert outcome.validation["flow_preflight_passed"] is True
    assert outcome.validation["metrics"]["healthy_motion_rows"] == 20


def test_flow_preflight_sequence_never_starts_telemetry_after_failed_deck_check(
    tmp_path: Path,
) -> None:
    process = importlib.import_module("flightrl.hardware.flow_preflight_process")
    marker = tmp_path / "telemetry-started"
    log_path = tmp_path / "telemetry.log"
    log_path.write_text("")

    outcome = process.run_flow_preflight_processes(
        deck_check_command=(sys.executable, "-c", "raise SystemExit(3)"),
        telemetry_command=_writer_command(marker, "started"),
        telemetry_path=tmp_path / "missing.csv",
        telemetry_log_path=log_path,
        deck_check_timeout_s=1.0,
        telemetry_timeout_s=1.0,
        cleanup_timeout_s=0.1,
        before_telemetry=lambda: None,
    )

    assert outcome.succeeded is False
    assert outcome.deck_check.returncode == 3
    assert outcome.telemetry is None
    assert outcome.validation is None
    assert marker.exists() is False


def test_flow_preflight_sequence_rejects_successful_child_with_unhealthy_csv(
    tmp_path: Path,
) -> None:
    process = importlib.import_module("flightrl.hardware.flow_preflight_process")
    csv_path = tmp_path / "flow.csv"
    log_path = tmp_path / "telemetry.log"
    log_path.write_text("")

    outcome = process.run_flow_preflight_processes(
        deck_check_command=(sys.executable, "-c", "pass"),
        telemetry_command=_writer_command(csv_path, _flow_csv(healthy=False)),
        telemetry_path=csv_path,
        telemetry_log_path=log_path,
        deck_check_timeout_s=1.0,
        telemetry_timeout_s=1.0,
        cleanup_timeout_s=0.1,
        before_telemetry=lambda: None,
    )

    assert outcome.succeeded is False
    assert outcome.validation is not None
    assert outcome.validation["flow_preflight_passed"] is False
    assert "flow_quality" in outcome.validation["failed_checks"]


def test_flow_preflight_cues_immediately_before_telemetry_child(tmp_path: Path) -> None:
    process = importlib.import_module("flightrl.hardware.flow_preflight_process")
    cue = tmp_path / "motion-start-cue"
    csv_path = tmp_path / "flow.csv"
    log_path = tmp_path / "telemetry.log"
    log_path.write_text("")
    telemetry = (
        sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            f"assert Path({str(cue)!r}).is_file(); "
            f"Path({str(csv_path)!r}).write_text({_flow_csv(healthy=True)!r})"
        ),
    )

    outcome = process.run_flow_preflight_processes(
        deck_check_command=(sys.executable, "-c", "pass"),
        telemetry_command=telemetry,
        telemetry_path=csv_path,
        telemetry_log_path=log_path,
        deck_check_timeout_s=1.0,
        telemetry_timeout_s=1.0,
        cleanup_timeout_s=0.1,
        before_telemetry=lambda: cue.write_text("start"),
    )

    assert outcome.succeeded is True


def test_flow_preflight_rejects_logged_packet_loss(tmp_path: Path) -> None:
    process = importlib.import_module("flightrl.hardware.flow_preflight_process")
    csv_path = tmp_path / "flow.csv"
    log_path = tmp_path / "telemetry.log"
    log_path.write_text("Too many packets lost\n")

    outcome = process.run_flow_preflight_processes(
        deck_check_command=(sys.executable, "-c", "pass"),
        telemetry_command=_writer_command(csv_path, _flow_csv(healthy=True)),
        telemetry_path=csv_path,
        telemetry_log_path=log_path,
        deck_check_timeout_s=1.0,
        telemetry_timeout_s=1.0,
        cleanup_timeout_s=0.1,
        before_telemetry=lambda: None,
    )

    assert outcome.succeeded is False
    assert outcome.packet_loss_free is False
    assert outcome.validation is None


def test_flow_preflight_dry_run_is_exact_non_actuating_and_writes_nothing(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "preflight"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_aideck_flow_preflight.py",
            "--config",
            str(CONFIG),
            "--run-dir",
            str(run_dir),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest["controls_drone"] is False
    assert manifest["non_actuating"] is True
    assert manifest["telemetry_uri"] == "usb://0"
    assert manifest["deck_expectations"] == {
        "expect_ai_deck": True,
        "expect_flow_deck": True,
        "expect_multiranger": False,
        "expect_zranger": True,
    }
    assert manifest["telemetry_variables"] == [
        "motion.motion",
        "motion.deltaX",
        "motion.deltaY",
        "motion.squal",
        "range.zrange",
    ]
    assert manifest["telemetry_period_ms"] == 50
    assert manifest["audible_cues"]["motion_start"].endswith("Glass.aiff")
    assert run_dir.exists() is False


def test_flow_preflight_dry_run_accepts_exact_radio_profile(tmp_path: Path) -> None:
    radio_config = tmp_path / "radio.toml"
    radio_config.write_text(
        CONFIG.read_text().replace(
            'uri = "usb://0"',
            'uri = "radio://0/80/2M/E7E7E7E7E7"',
        )
    )
    run_dir = tmp_path / "preflight"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_aideck_flow_preflight.py",
            "--config",
            str(radio_config),
            "--run-dir",
            str(run_dir),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest["telemetry_uri"] == "radio://0/80/2M/E7E7E7E7E7"
    assert manifest["controls_drone"] is False
    assert run_dir.exists() is False


def test_flow_preflight_rejects_wrong_deck_profile_before_spawning(
    tmp_path: Path,
) -> None:
    source = CONFIG.read_text()
    wrong_config = tmp_path / "wrong.toml"
    wrong_config.write_text(
        source.replace("expect_multiranger = false", "expect_multiranger = true")
    )
    run_dir = tmp_path / "preflight"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_aideck_flow_preflight.py",
            "--config",
            str(wrong_config),
            "--run-dir",
            str(run_dir),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "exact AI Deck, Flow Deck, Z-ranger, and no Multiranger" in result.stderr
    assert run_dir.exists() is False


def test_strict_stack_inspection_rejects_attached_multiranger() -> None:
    contract = importlib.import_module("flightrl.hardware.flow_preflight_contract")
    values = {
        "deck.bcAI": "1",
        "deck.bcFlow2": "1",
        "deck.bcMultiranger": "1",
        "deck.bcZRanger2": "1",
    }
    toc = {
        "motion": {"motion": object(), "deltaX": object(), "deltaY": object(), "squal": object()},
        "range": {"zrange": object()},
    }
    scf = SimpleNamespace(
        cf=SimpleNamespace(
            param=SimpleNamespace(get_value=values.__getitem__),
            log=SimpleNamespace(toc=SimpleNamespace(toc=toc)),
        )
    )

    with pytest.raises(HardwareSafetyError, match="deck.bcMultiranger"):
        contract.inspect_exact_flow_preflight_stack(scf)


def test_strict_stack_inspection_accepts_official_flow_motion_toc() -> None:
    contract = importlib.import_module("flightrl.hardware.flow_preflight_contract")
    values = {
        "deck.bcAI": "1",
        "deck.bcFlow2": "1",
        "deck.bcMultiranger": "0",
        "deck.bcZRanger2": "1",
    }
    toc = {
        "motion": {
            "motion": object(),
            "deltaX": object(),
            "deltaY": object(),
            "squal": object(),
        },
        "range": {"zrange": object()},
    }
    scf = SimpleNamespace(
        cf=SimpleNamespace(
            param=SimpleNamespace(get_value=values.__getitem__),
            log=SimpleNamespace(toc=SimpleNamespace(toc=toc)),
        )
    )

    report = contract.inspect_exact_flow_preflight_stack(scf)

    assert report["toc_variables"] == [
        "motion.motion",
        "motion.deltaX",
        "motion.deltaY",
        "motion.squal",
        "range.zrange",
    ]


def test_strict_stack_inspection_rejects_missing_required_toc_variable() -> None:
    contract = importlib.import_module("flightrl.hardware.flow_preflight_contract")
    values = {
        "deck.bcAI": "1",
        "deck.bcFlow2": "1",
        "deck.bcMultiranger": "0",
        "deck.bcZRanger2": "1",
    }
    toc = {
        "motion": {"motion": object(), "deltaX": object(), "deltaY": object()},
        "range": {"zrange": object()},
    }
    scf = SimpleNamespace(
        cf=SimpleNamespace(
            param=SimpleNamespace(get_value=values.__getitem__),
            log=SimpleNamespace(toc=SimpleNamespace(toc=toc)),
        )
    )

    with pytest.raises(HardwareSafetyError, match="motion.squal"):
        contract.inspect_exact_flow_preflight_stack(scf)
