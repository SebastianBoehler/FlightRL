from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from flightrl.mujoco.semantic_evaluation import evaluate_semantic_policy
from flightrl.mujoco.semantic_puffer_driver import SemanticPufferDriver
from flightrl.mujoco.semantic_reporting import shadow_gate_passed
from flightrl.mujoco.semantic_teacher_evaluation import evaluate_semantic_teacher
from flightrl.mujoco.semantic_vision_policy import SemanticVisionPolicy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run complete held-out evaluation for a semantic checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rooms", type=int, default=8)
    parser.add_argument("--agents-per-room", type=int, default=1)
    parser.add_argument("--steps", type=int, default=6_500)
    parser.add_argument("--room-profile", choices=("standard", "diverse"))
    args = parser.parse_args()

    report = json.loads(args.training_report.read_text())
    digest = _sha256(args.checkpoint)
    if report.get("checkpoint_sha256") != digest:
        raise ValueError("checkpoint does not match its training report")
    vision = report["observation_contract"]["vision"]
    seed = int(report["training_room_seeds"][0])
    room_profile = args.room_profile or report.get("eval_room_profile", "standard")
    def build_driver() -> SemanticPufferDriver:
        return SemanticPufferDriver(
            room_seeds=tuple(seed + 20_000 + index for index in range(args.rooms)),
            agents_per_room=args.agents_per_room,
            seed=seed + 20_000,
            active_exploration=bool(report["active_exploration"]),
            vision_width=int(vision["width"]),
            vision_height=int(vision["height"]),
            room_profile=room_profile,
        )

    driver = build_driver()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    policy = SemanticVisionPolicy(
        driver.driver_env,
        hidden_size=int(state["encoder.fusion.0.weight"].shape[0]),
        shared_visual_safety="clearance_head.weight" in state,
        recurrent_safety="recurrent_safety.clearance_head.weight" in state,
        recurrent_visual_safety=(
            "recurrent_visual_safety.clearance_head.weight" in state
        ),
    )
    policy.load_state_dict(state)
    driver.close()
    evaluations = {}
    for mode in ("full", "target_map_masked"):
        driver = build_driver()
        try:
            evaluations[mode] = evaluate_semantic_policy(
                policy,
                driver,
                steps=args.steps,
                mode=mode,
            )
        finally:
            driver.close()
    driver = build_driver()
    try:
        teacher_evaluation = evaluate_semantic_teacher(
            driver,
            steps=args.steps,
        )
    finally:
        driver.close()
    selected = "post_training_long"
    report["evaluation"][selected] = evaluations
    report["selected_stage"] = selected
    report["heldout_room_seeds"] = [
        seed + 20_000 + index for index in range(args.rooms)
    ]
    report["eval_room_profile"] = room_profile
    report["teacher_evaluation"] = teacher_evaluation
    report["post_training_evaluation"] = {
        "steps_per_agent": args.steps,
        "rooms": args.rooms,
        "agents_per_room": args.agents_per_room,
        "checkpoint_sha256": digest,
    }
    report["target_memory_is_causal"] = (
        evaluations["full"].get("success_rate", 0.0)
        > evaluations["target_map_masked"].get("success_rate", 0.0)
    )
    report["shadow_gate_passed"] = shadow_gate_passed(
        evaluations["full"],
        evaluations["target_map_masked"],
        active_exploration=bool(report["active_exploration"]),
        teacher=teacher_evaluation,
    )
    report["deployment_status"] = (
        "eligible for recorded-flight replay"
        if report["shadow_gate_passed"]
        else "shadow-only; complete held-out simulation gate failed"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evaluations["full"], indent=2, sort_keys=True))
    print(f"shadow_gate_passed={report['shadow_gate_passed']}")
    print(f"report={args.output}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
