from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flightrl.vision import VisionActionScale, load_aligned_vision_actions, load_vision_action_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay-gate a visual trajectory-imitation checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vision", action="append", required=True)
    parser.add_argument("--telemetry", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.vision) != len(args.telemetry):
        raise SystemExit("--vision and --telemetry must be supplied the same number of times")

    policy = load_vision_action_policy(args.checkpoint)
    scale = VisionActionScale(
        policy.metadata.velocity_scale_m_s,
        policy.metadata.yawrate_scale_deg_s,
    )
    dataset = load_aligned_vision_actions(args.vision, args.telemetry, scale=scale)
    if policy.metadata.contract_json and policy.metadata.contract_json != dataset.contract_json:
        raise SystemExit("checkpoint and replay vision contracts differ")
    with torch.no_grad():
        predicted = policy(torch.from_numpy(dataset.observations)).numpy()
    report = {
        "checkpoint": args.checkpoint,
        "samples": len(dataset.actions),
        "runs": len(np.unique(dataset.run_ids)),
        "max_alignment_error_ms": float(dataset.alignment_error_s.max() * 1000.0),
        "overall": metrics(predicted, dataset.actions, scale),
        "by_phase": {
            str(phase): metrics(predicted[dataset.phases == phase], dataset.actions[dataset.phases == phase], scale)
            for phase in np.unique(dataset.phases)
        },
        "controls_drone": False,
        "gate": "analysis_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def metrics(predicted: np.ndarray, expected: np.ndarray, scale: VisionActionScale) -> dict[str, float]:
    error = np.abs(scale.physical(predicted) - scale.physical(expected))
    return {
        "normalized_mse": float(np.mean((predicted - expected) ** 2)),
        "vx_mae_m_s": float(error[:, 0].mean()),
        "vy_mae_m_s": float(error[:, 1].mean()),
        "yawrate_mae_deg_s": float(error[:, 2].mean()),
        "vx_p95_m_s": float(np.percentile(error[:, 0], 95)),
        "vy_p95_m_s": float(np.percentile(error[:, 1], 95)),
        "yawrate_p95_deg_s": float(np.percentile(error[:, 2], 95)),
    }


if __name__ == "__main__":
    main()
