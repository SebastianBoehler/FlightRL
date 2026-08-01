from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from flightrl.hardware.aideck_stream import AiDeckUdpStream
from flightrl.semantic import (
    AsyncGroundingPipeline,
    ClipCropVerifier,
    ClipVerifierConfig,
    DiscoveryConfig,
    GroundingDinoConfig,
    GroundingDinoGrounder,
    SemanticRunWriter,
    collect_camera_only,
    require_semantic_frame,
)
from flightrl.semantic.bounded_authority import PufferBoundedAuthority
from flightrl.semantic.live import (
    SemanticFlightConfig,
    run_semantic_flight,
    write_summary,
)
from flightrl.semantic.puffer_shadow import SemanticPufferShadow
from flightrl.semantic.readiness import (
    load_bounded_forward_readiness,
    load_yaw_only_readiness,
)
from flightrl.semantic.yaw_authority import PufferYawAuthority


def main() -> None:
    args = parse_args()
    readiness = resolve_readiness(args)
    if args.dry_run:
        print_plan(args)
        return
    require_confirmations(args)

    output = Path(args.output or default_output(args.prompt))
    discovery = DiscoveryConfig(
        minimum_confidence=args.threshold,
        grounding_stale_s=args.grounding_stale_s,
        search_yawrate_deg_s=args.search_yawrate_deg_s,
        track_yawrate_deg_s=args.track_yawrate_deg_s,
        centered_hold_s=args.centered_hold_s,
        minimum_scan_s=args.minimum_scan_s,
        max_duration_s=args.duration_s,
        search_radius_m=args.search_radius_m,
        reposition_speed_m_s=args.reposition_speed_m_s,
        allow_reposition=args.confirm_bounded_exploration,
    )
    flight = SemanticFlightConfig(
        height_m=args.height_m,
        max_duration_s=args.duration_s,
        min_frame_width=args.min_frame_width,
        min_frame_mean=args.min_frame_mean,
        allow_reposition=args.confirm_bounded_exploration,
    )
    verifier = ClipCropVerifier(
        ClipVerifierConfig(
            model_id=args.verifier_model_id,
            device=args.device,
            minimum_probability=args.verifier_minimum_probability,
            minimum_margin=args.verifier_minimum_margin,
        )
    )
    grounder = GroundingDinoGrounder(
        GroundingDinoConfig(
            model_id=args.model_id,
            device=args.device,
            threshold=args.threshold,
        ),
        verifier=verifier,
    )
    stream = AiDeckUdpStream(
        host=args.aideck_host,
        port=args.aideck_port,
        bind_host=args.bind_host,
        bind_port=args.bind_port,
        timeout_s=args.camera_timeout_s,
    )
    pipeline = AsyncGroundingPipeline(stream, grounder, args.prompt)
    policy_shadow = (
        None
        if args.shadow_checkpoint is None
        else SemanticPufferShadow.from_training_report(
            args.shadow_checkpoint,
            args.shadow_training_report,
        )
    )
    policy_authority = None
    if args.puffer_yaw_control:
        policy_authority = PufferYawAuthority(policy_shadow, readiness)
    elif args.puffer_bounded_control:
        policy_authority = PufferBoundedAuthority(policy_shadow, readiness)
    manifest = {
        "prompt": args.prompt,
        "model_id": args.model_id,
        "device": args.device,
        "threshold": args.threshold,
        "distractor_labels": list(grounder.config.distractor_labels),
        "verifier_model_id": args.verifier_model_id,
        "verifier_minimum_probability": args.verifier_minimum_probability,
        "verifier_minimum_margin": args.verifier_minimum_margin,
        "controls_drone": args.flight,
        "bounded_exploration": args.confirm_bounded_exploration,
        "hardware_config": args.hardware_config,
        "shadow_checkpoint": args.shadow_checkpoint,
        "shadow_training_report": args.shadow_training_report,
        "policy_shadow_controls_drone": False,
        "policy_controls_drone": policy_authority is not None,
        "policy_authority": (
            policy_authority.approved_authority
            if policy_authority is not None
            else "none"
        ),
        "policy_axis_authority": (
            None if readiness is None else readiness["axis_authority"]
        ),
        "puffer_readiness_report": args.puffer_readiness_report,
    }
    try:
        pipeline.start()
        frame = pipeline.wait_for_frame(args.camera_timeout_s)
        require_semantic_frame(
            frame,
            min_width=flight.min_frame_width,
            min_mean=flight.min_frame_mean,
        )
        pipeline.wait_for_result(args.first_result_timeout_s)
        with SemanticRunWriter(output, manifest=manifest) as writer:
            if args.flight:
                summary = run_semantic_flight(
                    pipeline,
                    writer,
                    hardware_config_path=args.hardware_config,
                    flight=flight,
                    discovery=discovery,
                    policy_shadow=policy_shadow,
                    policy_authority=policy_authority,
                )
            else:
                summary = collect_camera_only(
                    pipeline,
                    writer,
                    duration_s=args.duration_s,
                    policy_shadow=policy_shadow,
                )
        summary.update(
            {
                "output": str(output),
                "camera_width": frame.width,
                "camera_height": frame.height,
                "camera_mean": float(frame.pixels.mean()),
            }
        )
        print(f"semantic run complete: {write_summary(output, summary)}")
    finally:
        pipeline.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find and face a text-prompted object using AI Deck frames"
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--hardware-config",
        default="configs/hardware/crazyflie_2_1_brushless_flow_only.toml",
    )
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument(
        "--verifier-model-id",
        default="openai/clip-vit-base-patch32",
    )
    parser.add_argument("--verifier-minimum-probability", type=float, default=0.60)
    parser.add_argument("--verifier-minimum-margin", type=float, default=0.45)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--height-m", type=float, default=0.3)
    parser.add_argument("--min-frame-width", type=int, default=128)
    parser.add_argument("--min-frame-mean", type=float, default=8.0)
    parser.add_argument("--grounding-stale-s", type=float, default=3.0)
    parser.add_argument("--search-yawrate-deg-s", type=float, default=20.0)
    parser.add_argument("--track-yawrate-deg-s", type=float, default=8.0)
    parser.add_argument("--centered-hold-s", type=float, default=1.0)
    parser.add_argument("--minimum-scan-s", type=float, default=0.0)
    parser.add_argument("--search-radius-m", type=float, default=0.25)
    parser.add_argument("--reposition-speed-m-s", type=float, default=0.06)
    parser.add_argument("--aideck-host", default="192.168.4.1")
    parser.add_argument("--aideck-port", type=int, default=5000)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--bind-port", type=int, default=5001)
    parser.add_argument("--camera-timeout-s", type=float, default=10.0)
    parser.add_argument("--first-result-timeout-s", type=float, default=15.0)
    parser.add_argument("--shadow-checkpoint")
    parser.add_argument("--shadow-training-report")
    parser.add_argument("--puffer-readiness-report")
    parser.add_argument("--puffer-yaw-control", action="store_true")
    parser.add_argument("--puffer-bounded-control", action="store_true")
    parser.add_argument("--flight", action="store_true")
    parser.add_argument("--confirm-flight", action="store_true")
    parser.add_argument("--confirm-semantic-yaw-control", action="store_true")
    parser.add_argument("--confirm-puffer-yaw-control", action="store_true")
    parser.add_argument("--confirm-puffer-bounded-control", action="store_true")
    parser.add_argument("--confirm-bounded-exploration", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_confirmations(args: argparse.Namespace) -> None:
    if (args.shadow_checkpoint is None) != (args.shadow_training_report is None):
        raise SystemExit(
            "--shadow-checkpoint and --shadow-training-report must be provided together"
        )
    if args.confirm_bounded_exploration and not args.flight:
        raise SystemExit("--confirm-bounded-exploration requires --flight")
    if args.puffer_yaw_control and args.puffer_bounded_control:
        raise SystemExit("choose exactly one Puffer control mode")
    if args.puffer_yaw_control or args.puffer_bounded_control:
        if not args.flight:
            raise SystemExit("Puffer control requires --flight")
        if args.shadow_checkpoint is None:
            raise SystemExit("Puffer control requires --shadow-checkpoint")
        if args.puffer_readiness_report is None:
            raise SystemExit("Puffer control requires --puffer-readiness-report")
        if args.confirm_bounded_exploration:
            raise SystemExit(
                "heuristic exploration is incompatible with Puffer authority"
            )
    if args.puffer_yaw_control:
        if not args.confirm_puffer_yaw_control:
            raise SystemExit(
                "--confirm-puffer-yaw-control is required for Puffer yaw authority"
            )
    if args.puffer_bounded_control and not args.confirm_puffer_bounded_control:
        raise SystemExit(
            "--confirm-puffer-bounded-control is required for Puffer translation"
        )
    if not args.flight:
        return
    if not args.confirm_flight:
        raise SystemExit("--confirm-flight is required for takeoff")
    if not args.confirm_semantic_yaw_control:
        raise SystemExit(
            "--confirm-semantic-yaw-control is required before detections control yaw"
        )


def resolve_readiness(args: argparse.Namespace) -> dict | None:
    if not (args.puffer_yaw_control or args.puffer_bounded_control):
        return None
    if args.shadow_checkpoint is None:
        raise SystemExit("--puffer-yaw-control requires --shadow-checkpoint")
    if args.puffer_readiness_report is None:
        raise SystemExit("--puffer-yaw-control requires --puffer-readiness-report")
    loader = (
        load_bounded_forward_readiness
        if args.puffer_bounded_control
        else load_yaw_only_readiness
    )
    return loader(args.puffer_readiness_report, args.shadow_checkpoint)


def print_plan(args: argparse.Namespace) -> None:
    mode = (
        "Puffer bounded forward+yaw"
        if args.puffer_bounded_control
        else (
            "Puffer yaw-only"
            if args.puffer_yaw_control
            else (
                "bounded reposition and yaw"
                if args.confirm_bounded_exploration
                else "heuristic yaw-only"
            )
        )
    )
    print(
        f"dry_run prompt={args.prompt!r} flight={args.flight} mode={mode} "
        f"duration_s={args.duration_s:.1f} height_m={args.height_m:.2f}"
    )


def default_output(prompt: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "".join(character if character.isalnum() else "-" for character in prompt)
    return f"artifacts/semantic/{stamp}-{slug.strip('-').lower()}"


if __name__ == "__main__":
    main()
