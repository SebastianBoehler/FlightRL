from __future__ import annotations

from pathlib import Path

from flightrl import load_config
from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_export import PUFFER4_NATIVE_FILES, export_puffer4_assets


ROOT = Path(__file__).resolve().parents[1]


def test_export_puffer4_assets_writes_binding_and_ini(tmp_path: Path) -> None:
    pufferlib_root = tmp_path / "PufferLib-4.0"
    (pufferlib_root / "config").mkdir(parents=True)
    (pufferlib_root / "ocean").mkdir(parents=True)

    config = load_config(ROOT / "configs" / "tasks" / "hover.toml")
    result = export_puffer4_assets(config, pufferlib_root)

    binding_path = result.env_dir / "binding.c"
    ini_text = result.config_path.read_text()
    binding_text = binding_path.read_text()

    assert result.env_name == "flightrl"
    assert binding_path.exists()
    assert "#define OBS_SIZE" in binding_text
    assert f"#define OBS_SIZE {config.observation_dim}" in binding_text
    assert f"#define NUM_ATNS {config.action_dim}" in binding_text
    assert 'dict_get(kwargs, "waypoint_7_z")' in binding_text
    assert "[base]" in ini_text
    assert "env_name = flightrl" in ini_text
    assert "total_agents = 256" in ini_text
    assert "replay_ratio = 2" in ini_text
    assert "horizon = 32" in ini_text
    assert "minibatch_size = 1024" in ini_text

    for filename in PUFFER4_NATIVE_FILES:
        assert (result.env_dir / filename).exists()


def test_export_puffer4_assets_respects_overrides(tmp_path: Path) -> None:
    pufferlib_root = tmp_path / "PufferLib-4.0"
    (pufferlib_root / "config").mkdir(parents=True)
    (pufferlib_root / "ocean").mkdir(parents=True)

    config = load_config(ROOT / "configs" / "tasks" / "reach.toml")
    settings = Puffer4ExportSettings(
        env_name="flightrl_reach",
        total_agents=384,
        num_buffers=6,
        num_threads=3,
        policy_hidden_size=192,
        policy_num_layers=4,
        train_seed=99,
    )
    result = export_puffer4_assets(config, pufferlib_root, settings=settings)
    ini_text = result.config_path.read_text()

    assert result.env_name == "flightrl_reach"
    assert "env_name = flightrl_reach" in ini_text
    assert "total_agents = 384" in ini_text
    assert "num_buffers = 6" in ini_text
    assert "num_threads = 3" in ini_text
    assert "hidden_size = 192" in ini_text
    assert "num_layers = 4" in ini_text
    assert "seed = 99" in ini_text
