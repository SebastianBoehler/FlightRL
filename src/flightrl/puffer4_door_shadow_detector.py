from __future__ import annotations

from flightrl.puffer4_door_shadow_detector_contract import (
    approved_shadow_detector_contract,
)
from flightrl.semantic import (
    ClipCropVerifier,
    ClipVerifierConfig,
    GroundingDinoConfig,
    GroundingDinoGrounder,
)


def approved_shadow_detector_configs(
    device: str,
) -> tuple[GroundingDinoConfig, ClipVerifierConfig]:
    contract = approved_shadow_detector_contract()
    clip = contract["clip_verifier"]
    grounding = contract["grounding_dino"]
    runtime_versions = tuple(sorted(contract["runtime_versions"].items()))
    clip_config = ClipVerifierConfig(
        model_id=clip["model_id"],
        revision=clip["revision"],
        artifact_manifest=tuple(sorted(clip["artifacts"].items())),
        runtime_versions=runtime_versions,
        weights_format=clip["weights_format"],
        device=device,
        minimum_probability=clip["minimum_probability"],
        minimum_margin=clip["minimum_margin"],
        crop_padding=clip["crop_padding"],
        negative_descriptions=tuple(clip["negative_descriptions"]),
    )
    grounding_config = GroundingDinoConfig(
        model_id=grounding["model_id"],
        revision=grounding["revision"],
        artifact_manifest=tuple(sorted(grounding["artifacts"].items())),
        runtime_versions=runtime_versions,
        weights_format=grounding["weights_format"],
        device=device,
        threshold=grounding["threshold"],
        autocontrast=grounding["autocontrast"],
        minimum_box_area=grounding["minimum_box_area"],
        maximum_box_area=grounding["maximum_box_area"],
        distractor_labels=tuple(grounding["distractor_labels"]),
    )
    return grounding_config, clip_config


def build_approved_shadow_grounder(device: str) -> GroundingDinoGrounder:
    grounding_config, clip_config = approved_shadow_detector_configs(device)
    verifier = ClipCropVerifier(clip_config)
    return GroundingDinoGrounder(
        grounding_config,
        verifier=verifier,
    )
