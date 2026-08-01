from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_door_bundle import load_fixed_door_checkpoint_bundle
from flightrl.puffer4_door_canonical_evaluation import (
    run_canonical_door_evaluation,
)
from flightrl.puffer4_door_challenge_evaluation import (
    resolve_canonical_output,
    validate_challenge_options,
)
from flightrl.puffer4_door_challenge_runner import (
    run_door_challenge_evaluation,
)
from flightrl.puffer4_door_challenge_specs import DOOR_CHALLENGE_NAMES
from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
)
from flightrl.puffer4_door_eval_provenance import (
    begin_fixed_door_evaluation_provenance,
)
from flightrl.puffer4_door_export import export_fixed_door_assets
from flightrl.puffer4_door_policy_contract import (
    verify_door_policy_contract,
)
from flightrl.puffer4_door_runner import (
    build_environment,
    load_puffer,
    verify_native_build,
)
from flightrl.puffer4_door_stream_contract import (
    door_stream_contract_report,
    verify_door_stream_contract,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUFFER = ROOT.parent / "PufferLib-4-flightrl"
DEFAULT_CHECKPOINT = (
    ROOT
    / "artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m"
    / "flightrl_fixed_door_d1_seed11_1048576.bin"
)


def main() -> None:
    provenance_capture = begin_fixed_door_evaluation_provenance(
        command=(sys.executable, *sys.argv),
        flightrl_root=ROOT,
        entrypoint=Path(__file__),
    )
    parser = argparse.ArgumentParser(
        description="Re-evaluate one fixed-door checkpoint without retraining"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--lineage-report",
        type=Path,
        required=True,
        help="hash-bound training or evaluation report for this checkpoint",
    )
    parser.add_argument("--puffer-root", type=Path, default=DEFAULT_PUFFER)
    parser.add_argument("--agents", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=10_011)
    parser.add_argument("--challenge", choices=DOOR_CHALLENGE_NAMES)
    parser.add_argument("--control-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--live-yaw-cap-challenge",
        action="store_true",
        help="separately evaluate the policy with yaw capped at the live limit",
    )
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    try:
        validate_challenge_options(
            challenge=args.challenge,
            control_report=args.control_report,
            output=args.output,
            live_yaw_cap_challenge=args.live_yaw_cap_challenge,
        )
    except ValueError as exc:
        parser.error(str(exc))
    bundle = load_fixed_door_checkpoint_bundle(
        args.checkpoint,
        args.lineage_report,
    )
    canonical_output = (
        resolve_canonical_output(
            bundle.checkpoint_path,
            lineage_report=bundle.report_path,
            requested=args.output,
        )
        if args.challenge is None
        else None
    )
    action_contract = bundle.action_contract
    architecture = bundle.architecture

    settings = Puffer4ExportSettings(
        env_name=bundle.env_name,
        total_agents=args.agents,
        num_buffers=1,
        num_threads=8,
        policy_hidden_size=architecture.hidden_size,
        policy_num_layers=architecture.num_layers,
        train_seed=bundle.train_seed,
    )
    export_fixed_door_assets(args.puffer_root, settings)
    if not args.skip_build:
        build_environment(args.puffer_root, bundle.env_name)
    native_build_fingerprint = verify_native_build(
        args.puffer_root,
        bundle.env_name,
    )
    puffer_args, torch_pufferl = load_puffer(
        args.puffer_root,
        bundle.env_name,
    )
    puffer_args["env"]["obstacle_probability"] = 0.0
    puffer_args["env"]["layout_diversity"] = 1.0
    action_contract.apply_to_env(puffer_args["env"])
    action_contract.verify_env(puffer_args["env"])
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT.verify_env(puffer_args["env"])
    puffer_args["vec"]["total_agents"] = args.agents
    vec = torch_pufferl._C.create_vec(puffer_args, torch_pufferl._C.gpu)
    try:
        policy = torch_pufferl.load_policy(puffer_args, vec)
        policy.load_state_dict(
            torch.load(
                bundle.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        policy_contract = bundle.policy_contract
        verify_door_policy_contract(
            policy_contract,
            hidden_size=int(policy.network.hidden_size),
            num_layers=int(policy.network.num_layers),
        )
        if vec.obs_size != policy_contract["observation"]["total_floats"]:
            raise ValueError("runtime observation size violates policy contract")
        stream_contract = door_stream_contract_report()
        verify_door_stream_contract(stream_contract)
    finally:
        vec.close()
    if args.challenge is not None:
        assert args.control_report is not None
        evaluation, output = run_door_challenge_evaluation(
            bundle=bundle,
            policy=policy,
            puffer_args=puffer_args,
            torch_pufferl=torch_pufferl,
            challenge=args.challenge,
            control_report=args.control_report,
            output=args.output,
            native_build_fingerprint=native_build_fingerprint,
            stream_contract=stream_contract,
            provenance_capture=provenance_capture,
            puffer_root=args.puffer_root,
            steps=args.steps,
            seed=args.seed,
            agents=args.agents,
        )
        print(json.dumps(evaluation, indent=2, sort_keys=True))
        print(f"output={output}")
        return
    assert canonical_output is not None
    evaluation, output = run_canonical_door_evaluation(
        bundle=bundle,
        policy=policy,
        puffer_args=puffer_args,
        torch_pufferl=torch_pufferl,
        output=canonical_output,
        native_build_fingerprint=native_build_fingerprint,
        stream_contract=stream_contract,
        provenance_capture=provenance_capture,
        puffer_root=args.puffer_root,
        steps=args.steps,
        seed=args.seed,
        agents=args.agents,
        live_yaw_cap_challenge=args.live_yaw_cap_challenge,
    )
    print(json.dumps(evaluation, indent=2, sort_keys=True))
    print(f"output={output}")


if __name__ == "__main__":
    main()
