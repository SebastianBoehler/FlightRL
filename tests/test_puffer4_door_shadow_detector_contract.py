from __future__ import annotations

import hashlib

from flightrl import puffer4_door_shadow_detector_contract as contract
from flightrl.puffer4_door_shadow_detector import (
    approved_shadow_detector_configs,
)


def test_detector_contract_pins_immutable_model_artifacts() -> None:
    detector = contract.approved_shadow_detector_contract()

    assert detector["contract_id"] == "fixed-door-real-shadow-detector-v3"
    assert detector["schema_version"] == 3
    assert detector["grounding_dino"]["revision"] == (
        "a2bb814dd30d776dcf7e30523b00659f4f141c71"
    )
    assert set(detector["grounding_dino"]["artifacts"]) == {
        "added_tokens.json",
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    }
    assert detector["grounding_dino"]["artifacts"]["model.safetensors"] == (
        "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
    )
    assert detector["grounding_dino"]["weights_format"] == "safetensors"
    assert detector["clip_verifier"]["revision"] == (
        "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    )
    assert set(detector["clip_verifier"]["artifacts"]) == {
        "config.json",
        "merges.txt",
        "preprocessor_config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    assert detector["clip_verifier"]["artifacts"]["pytorch_model.bin"] == (
        "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"
    )
    assert detector["clip_verifier"]["weights_format"] == "pytorch"
    assert detector["runtime_versions"]["transformers"] == "5.14.1"
    assert detector["runtime_versions"]["torch"] == "2.13.0"


def test_approved_detector_factory_derives_configs_from_contract() -> None:
    detector = contract.approved_shadow_detector_contract()

    grounding, clip = approved_shadow_detector_configs("cpu")

    assert dict(grounding.artifact_manifest) == detector["grounding_dino"][
        "artifacts"
    ]
    assert grounding.weights_format == detector["grounding_dino"][
        "weights_format"
    ]
    assert dict(clip.artifact_manifest) == detector["clip_verifier"]["artifacts"]
    assert clip.weights_format == detector["clip_verifier"]["weights_format"]
    assert dict(grounding.runtime_versions) == detector["runtime_versions"]
    assert clip.runtime_versions == grounding.runtime_versions


def test_hardware_config_snapshot_is_immune_to_source_mutation(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "hardware.toml"
    approved = b"[vehicle]\nuri = 'radio://approved'\n"
    source.write_bytes(approved)
    monkeypatch.setattr(
        contract,
        "APPROVED_SHADOW_HARDWARE_CONFIG",
        source.resolve(),
    )
    monkeypatch.setattr(
        contract,
        "APPROVED_SHADOW_HARDWARE_CONFIG_SHA256",
        hashlib.sha256(approved).hexdigest(),
    )

    with contract.approved_shadow_hardware_config_snapshot(source) as snapshot:
        source.write_text("[vehicle]\nuri = 'radio://changed'\n")

        assert snapshot != source.resolve()
        assert snapshot.read_bytes() == approved

    assert not snapshot.exists()
