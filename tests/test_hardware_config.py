from __future__ import annotations

from pathlib import Path

import pytest

from flightrl.hardware.config import DEFAULT_LOG_VARIABLES, load_hardware_config
from flightrl.hardware.errors import HardwareConfigError


ROOT = Path(__file__).resolve().parents[1]


def test_load_crazyflie_hardware_config() -> None:
    config = load_hardware_config(ROOT / "configs" / "hardware" / "crazyflie_2_1_brushless.toml")

    assert config.radio.uri.startswith("radio://")
    assert config.safety.default_height_m == pytest.approx(0.3)
    assert config.safety.requires_manual_confirm is True
    assert config.decks.expect_flow_deck is True
    assert config.decks.expect_multiranger is True
    assert config.decks.expect_ai_deck is False


def test_default_hardware_config_uses_comprehensive_log_profile() -> None:
    config = load_hardware_config(ROOT / "configs" / "hardware" / "crazyflie_2_1_brushless.toml")

    assert config.logging.variables == DEFAULT_LOG_VARIABLES
    assert len(config.logging.variables) <= 80
    assert "stateEstimate.qw" in config.logging.variables
    assert "controller.cmd_thrust" in config.logging.variables
    assert "motor.m1req" in config.logging.variables
    assert "rpm.m3" in config.logging.variables
    assert "kalman.varZ" in config.logging.variables
    assert config.logging.variable_types["rpm.m3"] == "uint16_t"
    assert config.logging.variable_types["motor.m1req"] == "int32_t"


def test_load_detailed_hardware_config_keeps_log_variable_types() -> None:
    config = load_hardware_config(ROOT / "configs" / "hardware" / "crazyflie_2_1_brushless_detailed.toml")

    assert "motor.m1" in config.logging.variables
    assert "controller.cmd_thrust" in config.logging.variables
    assert config.logging.variable_types["motor.m1"] == "uint16_t"
    assert config.logging.variable_types["motor.m1req"] == "int32_t"


def test_load_low_level_hardware_config_stays_under_cflib_block_budget() -> None:
    config = load_hardware_config(ROOT / "configs" / "hardware" / "crazyflie_2_1_brushless_low_level.toml")

    assert len(config.logging.variables) <= 80
    assert "rpm.m3" in config.logging.variables
    assert "health.motorVarXM3" in config.logging.variables
    assert "pid_rate.roll_outP" in config.logging.variables
    assert config.logging.variable_types["rpm.m3"] == "uint16_t"


def test_load_flow_only_hardware_config_does_not_require_multiranger() -> None:
    config = load_hardware_config(ROOT / "configs" / "hardware" / "crazyflie_2_1_brushless_flow_only.toml")

    assert config.decks.expect_flow_deck is True
    assert config.decks.expect_multiranger is False
    assert "stateEstimate.x" in config.logging.variables
    assert "supervisor.info" in config.logging.variables
    assert "sys.canfly" in config.logging.variables
    assert "range.front" not in config.logging.variables
    assert len(config.logging.variables) <= 80


def test_load_aideck_flow2_profile_binds_exact_deck_stack() -> None:
    config = load_hardware_config(
        ROOT
        / "configs"
        / "hardware"
        / "crazyflie_2_1_brushless_aideck_flow2.toml"
    )

    assert config.decks.expect_ai_deck is True
    assert config.decks.expect_flow_deck is True
    assert config.decks.expect_zranger is True
    assert config.decks.expect_multiranger is False


def test_load_aideck_usb_capture_profile_uses_one_telemetry_block() -> None:
    config = load_hardware_config(
        ROOT
        / "configs"
        / "hardware"
        / "crazyflie_2_1_brushless_aideck_flow2_usb_capture.toml"
    )

    assert config.radio.uri == "usb://0"
    assert config.decks.expect_ai_deck is True
    assert config.decks.expect_flow_deck is True
    assert config.decks.expect_zranger is True
    assert config.logging.variables == (
        "stateEstimate.x",
        "stateEstimate.y",
        "stateEstimate.z",
        "stateEstimate.yaw",
        "pm.vbat",
    )


def test_load_aideck_usb_flow_preflight_profile_uses_raw_motion_block() -> None:
    config = load_hardware_config(
        ROOT
        / "configs"
        / "hardware"
        / "crazyflie_2_1_brushless_aideck_flow2_usb_flow_preflight.toml"
    )

    assert config.radio.uri == "usb://0"
    assert config.logging.variables == (
        "motion.motion",
        "motion.deltaX",
        "motion.deltaY",
        "motion.squal",
        "range.zrange",
    )


def test_load_aideck_radio_flow_preflight_profile_uses_exact_uri() -> None:
    config = load_hardware_config(
        ROOT
        / "configs"
        / "hardware"
        / "crazyflie_2_1_brushless_aideck_flow2_radio_flow_preflight.toml"
    )

    assert config.radio.uri == "radio://0/80/2M/E7E7E7E7E7"
    assert config.decks.expect_ai_deck is True
    assert config.decks.expect_flow_deck is True
    assert config.decks.expect_zranger is True
    assert config.logging.variables == (
        "motion.motion",
        "motion.deltaX",
        "motion.deltaY",
        "motion.squal",
        "range.zrange",
    )


def test_invalid_demo_height_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        """
[radio]
uri = "radio://0/80/2M/E7E7E7E7E7"

[safety]
default_height_m = 1.5
""".strip()
    )

    with pytest.raises(HardwareConfigError, match="default_height_m"):
        load_hardware_config(path)
