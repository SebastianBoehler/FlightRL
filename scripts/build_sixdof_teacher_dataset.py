from __future__ import annotations

import argparse
import json

from flightrl.sixdof.dataset import collect_teacher_dataset, merge_datasets, write_dataset
from flightrl.sixdof.observation import OBSERVATION_MODES


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline 6-DoF teacher-rollout imitation dataset")
    parser.add_argument("--task", default="position_yaw,obstacle_avoidance,circle")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--native-step", action="store_true")
    parser.add_argument("--reset-profile", default=None, help="Named reset curriculum, for example position_yaw_easy or position_yaw_medium.")
    parser.add_argument("--observation-mode", default="base", choices=OBSERVATION_MODES)
    parser.add_argument("--append-dataset", action="append", default=[], help="Existing compatible dataset to prepend.")
    parser.add_argument("--output", default="artifacts/datasets/sixdof_teacher_safe_tasks.npz")
    args = parser.parse_args()

    dataset = collect_teacher_dataset(
        task_spec=args.task,
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
        use_native_step=args.native_step,
        reset_profile=args.reset_profile,
        observation_mode=args.observation_mode,
    )
    if args.append_dataset:
        dataset = merge_datasets(args.append_dataset, dataset)
    output = write_dataset(args.output, dataset)
    print(f"dataset={output}")
    print(json.dumps(dataset["metadata"], sort_keys=True))


if __name__ == "__main__":
    main()
