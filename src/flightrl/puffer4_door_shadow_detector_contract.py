from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import json
from math import isclose
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Any, Mapping

from flightrl.semantic.clip_verifier import DEFAULT_NEGATIVE_DESCRIPTIONS


ROOT = Path(__file__).resolve().parents[2]
APPROVED_SHADOW_PROMPT = "interior door"
APPROVED_SHADOW_DETECTOR_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
APPROVED_SHADOW_DETECTOR_REVISION = (
    "a2bb814dd30d776dcf7e30523b00659f4f141c71"
)
APPROVED_SHADOW_DETECTOR_ARTIFACTS = MappingProxyType(
    {
        "added_tokens.json": (
            "909e96cb32d92ce728a01bc99850cbba26196d74115c17ebeb019275412588f2"
        ),
        "config.json": (
            "eec82c5ab66e16df12a9a212e68ac011779927c2536cf9078658e35d85f0c67a"
        ),
        "model.safetensors": (
            "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
        ),
        "preprocessor_config.json": (
            "8454179ba95e2ad22947835aad7b45862a601fc0055ab88bf1ee70892d3aea60"
        ),
        "special_tokens_map.json": (
            "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3"
        ),
        "tokenizer.json": (
            "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
        ),
        "tokenizer_config.json": (
            "d40ab645b68211910b9170d22433d43186a6ec8ee6fd10ba170524b25bf4fb56"
        ),
        "vocab.txt": (
            "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
        ),
    }
)
APPROVED_SHADOW_THRESHOLD = 0.25
APPROVED_SHADOW_DEVICE = "mps"
APPROVED_SHADOW_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
APPROVED_SHADOW_CLIP_REVISION = (
    "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
)
APPROVED_SHADOW_CLIP_ARTIFACTS = MappingProxyType(
    {
        "config.json": (
            "b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770"
        ),
        "merges.txt": (
            "f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85"
        ),
        "preprocessor_config.json": (
            "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd"
        ),
        "pytorch_model.bin": (
            "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"
        ),
        "special_tokens_map.json": (
            "f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf"
        ),
        "tokenizer.json": (
            "b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842"
        ),
        "tokenizer_config.json": (
            "34b7336e4bee12e0a9730eaf5189f582ef3c3eea5027f65730e5717256755aad"
        ),
        "vocab.json": (
            "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363"
        ),
    }
)
APPROVED_SHADOW_RUNTIME_VERSIONS = MappingProxyType(
    {
        "Pillow": "12.3.0",
        "huggingface-hub": "1.24.0",
        "numpy": "2.5.1",
        "safetensors": "0.8.0",
        "tokenizers": "0.22.2",
        "torch": "2.13.0",
        "transformers": "5.14.1",
    }
)
APPROVED_SHADOW_HARDWARE_CONFIG = (
    ROOT / "configs/hardware/crazyflie_2_1_brushless_flow_only.toml"
).resolve()
APPROVED_SHADOW_HARDWARE_CONFIG_SHA256 = (
    "7fe3f7c45b91982c5ef734d9ef9bcd8c392b4193db0f7bca697d777aef819e73"
)


def approved_shadow_detector_contract() -> dict[str, Any]:
    payload = {
        "contract_id": "fixed-door-real-shadow-detector-v3",
        "schema_version": 3,
        "prompt": APPROVED_SHADOW_PROMPT,
        "runtime_versions": dict(APPROVED_SHADOW_RUNTIME_VERSIONS),
        "grounding_dino": {
            "model_id": APPROVED_SHADOW_DETECTOR_MODEL_ID,
            "revision": APPROVED_SHADOW_DETECTOR_REVISION,
            "artifacts": dict(APPROVED_SHADOW_DETECTOR_ARTIFACTS),
            "weights_format": "safetensors",
            "threshold": APPROVED_SHADOW_THRESHOLD,
            "autocontrast": True,
            "minimum_box_area": 0.0005,
            "maximum_box_area": 0.5,
            "distractor_labels": [],
        },
        "clip_verifier": {
            "model_id": APPROVED_SHADOW_CLIP_MODEL_ID,
            "revision": APPROVED_SHADOW_CLIP_REVISION,
            "artifacts": dict(APPROVED_SHADOW_CLIP_ARTIFACTS),
            "weights_format": "pytorch",
            "minimum_probability": 0.60,
            "minimum_margin": 0.45,
            "crop_padding": 0.6,
            "negative_descriptions": list(DEFAULT_NEGATIVE_DESCRIPTIONS),
        },
        "preprocessing": {
            "grounder_self_mask": (
                "normalized-airframe-mask-global-channel-mean-fill-v1"
            ),
            "detector_image": "RGB-autocontrast-v1",
            "policy_frame": "policy-contract-observation-frame-v1",
        },
    }
    return payload | {"sha256": _payload_sha256(payload)}


def approved_shadow_hardware_config_identity() -> dict[str, str]:
    return {
        "path": str(APPROVED_SHADOW_HARDWARE_CONFIG),
        "sha256": APPROVED_SHADOW_HARDWARE_CONFIG_SHA256,
    }


@contextmanager
def approved_shadow_hardware_config_snapshot(
    hardware_config: str | Path,
) -> Iterator[Path]:
    """Yield a private immutable-content copy of the approved live config."""
    config_path = Path(hardware_config).resolve()
    if config_path != APPROVED_SHADOW_HARDWARE_CONFIG:
        raise ValueError("real shadow requires the approved hardware config")
    try:
        data = config_path.read_bytes()
    except OSError as exc:
        raise ValueError("approved shadow hardware config is unreadable") from exc
    if hashlib.sha256(data).hexdigest() != APPROVED_SHADOW_HARDWARE_CONFIG_SHA256:
        raise ValueError("real shadow requires the approved hardware config")
    with NamedTemporaryFile(suffix=".toml") as snapshot:
        snapshot.write(data)
        snapshot.flush()
        yield Path(snapshot.name)


def require_approved_shadow_runtime(
    *,
    prompt: str,
    detector_model_id: str,
    threshold: float,
    device: str,
    hardware_config: str | Path,
) -> None:
    config_path = Path(hardware_config).resolve()
    if (
        prompt != APPROVED_SHADOW_PROMPT
        or detector_model_id != APPROVED_SHADOW_DETECTOR_MODEL_ID
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not isclose(
            float(threshold),
            APPROVED_SHADOW_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or device != APPROVED_SHADOW_DEVICE
        or config_path != APPROVED_SHADOW_HARDWARE_CONFIG
    ):
        raise ValueError(
            "real shadow requires the approved shadow detector runtime"
        )
    if _file_sha256(config_path) != APPROVED_SHADOW_HARDWARE_CONFIG_SHA256:
        raise ValueError(
            "real shadow requires the approved shadow detector runtime"
        )


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError("approved shadow hardware config is unreadable") from exc
    return digest.hexdigest()
