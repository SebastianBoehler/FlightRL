from __future__ import annotations

from pathlib import Path

import pytest

from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.errors import HardwareConfigError


ROOT = Path(__file__).resolve().parents[1]


def test_load_crazyflie_hardware_config() -> None:
    config = load_hardware_config(ROOT / "configs" / "hardware" / "crazyflie_2_1_brushless.toml")

    assert config.radio.uri.startswith("radio://")
    assert config.safety.default_height_m == pytest.approx(0.3)
    assert config.safety.requires_manual_confirm is True
    assert config.decks.expect_flow_deck is True
    assert config.decks.expect_multiranger is True


def test_load_detailed_hardware_config_keeps_log_variable_types() -> None:
    config = load_hardware_config(ROOT / "configs" / "hardware" / "crazyflie_2_1_brushless_detailed.toml")

    assert "motor.m1" in config.logging.variables
    assert "controller.cmd_thrust" in config.logging.variables
    assert config.logging.variable_types["motor.m1"] == "uint16_t"
    assert config.logging.variable_types["motor.m1req"] == "int32_t"


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
