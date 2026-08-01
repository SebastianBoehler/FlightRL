from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from flightrl.hardware.aideck_stream import AiDeckUdpStream
from flightrl.puffer4_door_control import (
    DoorPufferControlAdapter,
    load_readiness_bound_control_adapter,
    require_readiness_bound_control_evidence,
)
from flightrl.puffer4_door_contract import (
    FIXED_DOOR_LIVE_SAFETY_CONTRACT,
)
from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
)
from flightrl.puffer4_door_readiness import load_fixed_door_yaw_readiness
from flightrl.puffer4_door_shadow_detector import (
    build_approved_shadow_grounder,
)
from flightrl.puffer4_door_shadow_detector_contract import (
    APPROVED_SHADOW_DETECTOR_MODEL_ID,
    APPROVED_SHADOW_DEVICE,
    APPROVED_SHADOW_HARDWARE_CONFIG,
    APPROVED_SHADOW_PROMPT,
    APPROVED_SHADOW_THRESHOLD,
    approved_shadow_detector_contract,
    approved_shadow_hardware_config_snapshot,
    require_approved_shadow_runtime,
)
from flightrl.semantic.readiness import file_sha256
from flightrl.puffer4_door_self_mask import DoorSelfMaskedGrounder
from flightrl.semantic import (
    AsyncGroundingPipeline,
    DiscoveryConfig,
    SemanticFlightConfig,
    SemanticRunWriter,
    require_semantic_frame,
    run_semantic_flight,
    write_summary,
)
from flightrl.semantic.yaw_authority import PufferYawAuthority


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT
    / "artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m"
    / "flightrl_fixed_door_d1_seed11_1048576.bin"
)
DEFAULT_EVALUATION = DEFAULT_CHECKPOINT.with_suffix(".reevaluation.json")
DEFAULT_READINESS = DEFAULT_CHECKPOINT.with_suffix(".yaw_readiness.json")


def main() -> None:
    args = parse_args()
    if args.dry_run:
        policy = DoorPufferControlAdapter.from_evaluation_report(
            args.checkpoint,
            args.evaluation_report,
        )
        print(json.dumps(dry_run(policy, args), indent=2, sort_keys=True))
        return
    live_safety = FIXED_DOOR_LIVE_SAFETY_CONTRACT
    live_safety.require_live_envelope(
        height_m=args.height_m,
        duration_s=args.duration_s,
    )
    require_confirmation(args)
    require_approved_shadow_runtime(
        prompt=APPROVED_SHADOW_PROMPT,
        detector_model_id=args.model_id,
        threshold=args.threshold,
        device=args.device,
        hardware_config=args.hardware_config,
    )
    readiness = load_fixed_door_yaw_readiness(
        args.readiness_report,
        args.checkpoint,
        args.evaluation_report,
    )
    policy = load_readiness_bound_control_adapter(
        args.checkpoint,
        args.evaluation_report,
        readiness,
    )
    limits = readiness["limits"]
    authority = PufferYawAuthority(policy, readiness)
    detector = approved_shadow_detector_contract()
    base_grounder = build_approved_shadow_grounder(args.device)
    stream = AiDeckUdpStream(
        host=args.aideck_host,
        port=args.aideck_port,
        bind_host=args.bind_host,
        bind_port=args.bind_port,
        timeout_s=args.camera_timeout_s,
    )
    pipeline = AsyncGroundingPipeline(
        stream,
        DoorSelfMaskedGrounder(base_grounder),
        APPROVED_SHADOW_PROMPT,
    )
    flight = SemanticFlightConfig(
        height_m=args.height_m,
        max_duration_s=args.duration_s,
        min_frame_width=128,
        min_frame_mean=8.0,
    )
    discovery = DiscoveryConfig(
        minimum_confidence=args.threshold,
        grounding_stale_s=(
            FIXED_DOOR_EVIDENCE_AGE_CONTRACT.maximum_evidence_age_s
        ),
        search_yawrate_deg_s=float(limits["search_abs_yawrate_deg_s"]),
        track_yawrate_deg_s=float(limits["detected_abs_yawrate_deg_s"]),
        centered_hold_s=1.0,
        max_duration_s=args.duration_s,
        allow_reposition=False,
    )
    output = args.output or default_output()
    manifest = {
        "checkpoint": str(policy.bundle.checkpoint_path),
        "checkpoint_sha256": policy.bundle.checkpoint_sha256,
        "evaluation_report": str(args.evaluation_report.resolve()),
        "evaluation_report_sha256": policy.bundle.report_sha256,
        "readiness_report": str(args.readiness_report.resolve()),
        "readiness_report_sha256": file_sha256(args.readiness_report),
        "prompt": APPROVED_SHADOW_PROMPT,
        "detector_contract": detector,
        "controls_drone": True,
        "policy": policy.bundle.action_contract.contract_id,
        "policy_authority": "yaw_only",
        "axis_authority": readiness["axis_authority"],
        "live_safety_contract": live_safety.to_report(),
        "firmware_stabilization": True,
        "maximum_yawrate_deg_s": float(limits["detected_abs_yawrate_deg_s"]),
        "height_m": args.height_m,
        "max_duration_s": args.duration_s,
        "translation_enabled": False,
    }
    try:
        pipeline.start()
        frame = pipeline.wait_for_frame(args.camera_timeout_s)
        require_semantic_frame(frame, min_width=128, min_mean=8.0)
        pipeline.wait_for_result(args.first_result_timeout_s)
        with SemanticRunWriter(output, manifest=manifest) as writer:
            with approved_shadow_hardware_config_snapshot(
                args.hardware_config
            ) as hardware_snapshot:
                require_readiness_bound_control_evidence(
                    args.checkpoint,
                    args.evaluation_report,
                    readiness,
                )
                summary = run_semantic_flight(
                    pipeline,
                    writer,
                    hardware_config_path=hardware_snapshot,
                    flight=flight,
                    discovery=discovery,
                    policy_authority=authority,
                )
        summary["output"] = str(output)
        summary["checkpoint"] = str(args.checkpoint.resolve())
        print(f"fixed-door run complete: {write_summary(output, summary)}")
    finally:
        pipeline.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bundled fixed-door student with bounded yaw authority"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=DEFAULT_EVALUATION,
    )
    parser.add_argument(
        "--readiness-report",
        type=Path,
        default=DEFAULT_READINESS,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--hardware-config",
        type=Path,
        default=APPROVED_SHADOW_HARDWARE_CONFIG,
    )
    parser.add_argument("--duration-s", type=float, default=15.0)
    parser.add_argument("--height-m", type=float, default=0.5)
    parser.add_argument(
        "--model-id",
        default=APPROVED_SHADOW_DETECTOR_MODEL_ID,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps"),
        default=APPROVED_SHADOW_DEVICE,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=APPROVED_SHADOW_THRESHOLD,
    )
    parser.add_argument("--aideck-host", default="192.168.4.1")
    parser.add_argument("--aideck-port", type=int, default=5000)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--bind-port", type=int, default=5001)
    parser.add_argument("--camera-timeout-s", type=float, default=10.0)
    parser.add_argument("--first-result-timeout-s", type=float, default=15.0)
    parser.add_argument("--confirm-flight", action="store_true")
    parser.add_argument("--confirm-fixed-door-yaw-control", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_confirmation(args: argparse.Namespace) -> None:
    if not args.confirm_flight:
        raise SystemExit("--confirm-flight is required for takeoff")
    if not args.confirm_fixed_door_yaw_control:
        raise SystemExit(
            "--confirm-fixed-door-yaw-control is required for learned yaw"
        )


def dry_run(policy: DoorPufferControlAdapter, args) -> dict:
    telemetry = {
        "stateEstimate.x": 0.0,
        "stateEstimate.y": 0.0,
        "stateEstimate.z": args.height_m,
        "stateEstimate.vx": 0.0,
        "stateEstimate.vy": 0.0,
        "stateEstimate.vz": 0.0,
        "stateEstimate.roll": 0.0,
        "stateEstimate.pitch": 0.0,
        "stateEstimate.yaw": 0.0,
        "gyro.x": 0.0,
        "gyro.y": 0.0,
        "gyro.z": 0.0,
    }
    proposal = policy.step(
        frame=np.full((96, 128), 51, dtype=np.uint8),
        telemetry=telemetry,
        prompt=APPROVED_SHADOW_PROMPT,
        detection=None,
    )
    return {
        "checkpoint": str(args.checkpoint.resolve()),
        "controls_drone": False,
        "readiness_exists": args.readiness_report.is_file(),
        "proposed_action": proposal,
        "next_step": "collect a matching real fixed-door shadow trace",
    }


def default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "artifacts/semantic" / f"{stamp}-fixed-door-yaw"


if __name__ == "__main__":
    main()
