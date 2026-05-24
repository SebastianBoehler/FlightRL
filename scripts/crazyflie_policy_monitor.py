from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import time

import torch

from flightrl import load_config
from flightrl.hardware.cflib_bridge import require_cflib, sync_crazyflie_context
from flightrl.hardware.policy_observation import (
    POLICY_LOG_VARIABLES,
    build_policy_observation,
    initial_policy_state,
    update_previous_action,
)
from flightrl.hardware.telemetry import build_log_configs
from flightrl.policy import create_policy_for_checkpoint
from flightrl.sim2real.hardware_approval import hardware_approval_status
from flightrl.training import create_env_and_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a FlightRL checkpoint against live Crazyflie telemetry without control")
    parser.add_argument("--config", default="configs/tasks/crazyflie_hover.toml")
    parser.add_argument("--hardware-config", default="configs/hardware/crazyflie_2_1_brushless.toml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--output", default="artifacts/crazyflie_logs/policy_monitor.csv")
    parser.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.45], metavar=("X", "Y", "Z"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--approval-manifest", default="artifacts/replay/sim2real_checkpoint_manifest_current_2026-05-20.json")
    args = parser.parse_args()

    approval = hardware_approval_status(args.checkpoint, args.approval_manifest)
    config = load_config(args.config)
    env, _unused = create_env_and_policy(config, policy_hidden_size=config.training.hidden_size)
    policy = create_policy_for_checkpoint(env, args.checkpoint, hidden_size=config.training.hidden_size)
    policy.eval()
    rows = _run_dry(config, policy, args.target, approval) if args.dry_run else _run_live(config, policy, args, approval)
    env.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]) if rows else ["host_time_s"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    print(f"monitor_only=True hardware_approved={approval['hardware_approved']} approval_status={approval['approval_status']}")


def _run_dry(config, policy, target, approval):
    state = initial_policy_state(config)
    telemetry = {"stateEstimate.z": target[2], "pm.vbat": 3.8}
    obs = build_policy_observation(config, telemetry, state, target=target)
    with torch.no_grad():
        logits, _value, _state = policy.forward_eval(torch.from_numpy(obs).view(1, -1))
    action = logits.mean.squeeze(0).numpy()
    update_previous_action(state, action)
    return [{"host_time_s": time(), "action_0": float(action[0]), "action_1": float(action[1]), **approval_columns(approval)}]


def _run_live(config, policy, args, approval):
    from flightrl.hardware.config import load_hardware_config

    hardware_config = load_hardware_config(args.hardware_config)
    hardware_config.logging.variables = POLICY_LOG_VARIABLES
    modules = require_cflib()
    state = initial_policy_state(config)
    latest: dict[str, float] = {}
    rows = []
    deadline = time() + args.duration_s
    with sync_crazyflie_context(hardware_config, modules) as scf:
        with modules.sync_logger_cls(scf, build_log_configs(modules, hardware_config)) as logger:
            while time() < deadline:
                _timestamp, values, _conf = next(logger)
                latest.update({key: float(value) for key, value in values.items()})
                obs = build_policy_observation(config, latest, state, target=args.target)
                with torch.no_grad():
                    logits, _value, _state = policy.forward_eval(torch.from_numpy(obs).view(1, -1))
                action = logits.mean.squeeze(0).numpy()
                update_previous_action(state, action)
                rows.append({"host_time_s": time(), "action_0": float(action[0]), "action_1": float(action[1]), **latest, **approval_columns(approval)})
    return rows


def approval_columns(approval: dict) -> dict[str, str | bool]:
    return {
        "monitor_only": True,
        "hardware_approved": bool(approval["hardware_approved"]),
        "approval_status": str(approval["approval_status"]),
        "approval_error": str(approval.get("approval_error", "")),
        "approval_task": str(approval.get("approval_task", "")),
        "approval_label": str(approval.get("approval_label", "")),
    }


if __name__ == "__main__":
    main()
