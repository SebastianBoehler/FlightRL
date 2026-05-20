from __future__ import annotations

import argparse
import json

from flightrl.sixdof.dagger import collect_policy_dataset, merge_datasets
from flightrl.sixdof.dataset import write_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect 6-DoF DAgger data from policy-visited states")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default=None, help="Optional task subset. Defaults to checkpoint tasks.")
    parser.add_argument("--append-dataset", action="append", default=[], help="Existing compatible dataset to prepend.")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=311)
    parser.add_argument("--beta", type=float, default=0.0, help="Teacher action mixing during rollout, 0=policy only.")
    parser.add_argument("--native-step", action="store_true")
    parser.add_argument("--reset-profile", default=None, help="Named reset curriculum for policy-state collection.")
    args = parser.parse_args()

    dataset = collect_policy_dataset(
        checkpoint_path=args.checkpoint,
        task_spec=args.task,
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
        use_native_step=args.native_step,
        beta=args.beta,
        reset_profile=args.reset_profile,
    )
    if args.append_dataset:
        dataset = merge_datasets(args.append_dataset, dataset)
    output = write_dataset(args.output, dataset)
    print(f"dataset={output}")
    print(json.dumps(dataset["metadata"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
