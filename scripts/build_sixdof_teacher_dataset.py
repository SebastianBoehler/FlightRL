from __future__ import annotations

import argparse
import json

from flightrl.sixdof.dataset import collect_teacher_dataset, write_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline 6-DoF teacher-rollout imitation dataset")
    parser.add_argument("--task", default="position_yaw,obstacle_avoidance,circle")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--native-step", action="store_true")
    parser.add_argument("--output", default="artifacts/datasets/sixdof_teacher_safe_tasks.npz")
    args = parser.parse_args()

    dataset = collect_teacher_dataset(
        task_spec=args.task,
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
        use_native_step=args.native_step,
    )
    output = write_dataset(args.output, dataset)
    print(f"dataset={output}")
    print(json.dumps(dataset["metadata"], sort_keys=True))


if __name__ == "__main__":
    main()
