from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
from time import perf_counter

import torch

from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_vision_export import export_visual_puffer4_assets
from flightrl.puffer4_vision_policy import FlightRLVisionEncoder, VISION_CHANNELS, VISION_PIXELS


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUFFER_ROOT = REPO_ROOT.parent / "PufferLib-4-flightrl"


def build_puffer_environment(puffer_root: Path, env_name: str) -> None:
    env = os.environ.copy()
    if sys.platform == "darwin":
        llvm = Path("/opt/homebrew/opt/llvm/bin")
        env.update({"CC": str(llvm / "clang"), "CXX": str(llvm / "clang++")})
    subprocess.run(
        ["bash", "build.sh", env_name, "--cpu"],
        cwd=puffer_root,
        env=env,
        check=True,
    )
    if sys.platform == "darwin":
        align_openmp_runtime(puffer_root)


def align_openmp_runtime(puffer_root: Path) -> None:
    torch_openmp = Path(torch.__file__).resolve().parent / "lib" / "libomp.dylib"
    extension = puffer_root / "pufferlib" / f"_C{sysconfig.get_config_var('EXT_SUFFIX')}"
    dependencies = subprocess.run(
        ["otool", "-L", str(extension)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for line in dependencies.splitlines():
        dependency = line.strip().split(" ", 1)[0]
        if dependency.endswith("/libomp.dylib") and Path(dependency) != torch_openmp:
            subprocess.run(
                ["install_name_tool", "-change", dependency, str(torch_openmp), str(extension)],
                check=True,
            )


def load_puffer(puffer_root: Path, env_name: str):
    sys.path.insert(0, str(puffer_root))
    from pufferlib import models, pufferl, torch_pufferl

    models.FlightRLVisionEncoder = FlightRLVisionEncoder
    old_argv = sys.argv
    try:
        sys.argv = ["train_puffer_visual_navigation"]
        args = pufferl.load_config(env_name)
    finally:
        sys.argv = old_argv
    args["world_size"] = 1
    return args, torch_pufferl


def train(
    args: dict,
    torch_pufferl,
    total_timesteps: int,
    log_interval: int,
    action_logstd: float,
    reset_action_head: bool,
):
    args["train"]["total_timesteps"] = total_timesteps
    vec = torch_pufferl._C.create_vec(args, torch_pufferl._C.gpu)
    policy = torch_pufferl.load_policy(args, vec)
    with torch.no_grad():
        policy.decoder.decoder_logstd.fill_(action_logstd)
        if reset_action_head:
            policy.decoder.decoder_mean.weight.zero_()
            policy.decoder.decoder_mean.bias.zero_()
    trainer = torch_pufferl.PuffeRL(args, vec, policy, verbose=False)
    epochs = max(1, total_timesteps // trainer.batch_size)
    history = []
    pending_env = []
    best_score = float("-inf")
    best_epoch = 0
    best_state = None
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        trainer.rollouts()
        if trainer.env_logs:
            pending_env.append(trainer.env_logs)
        trainer.train()
        if epoch == 1 or epoch == epochs or epoch % log_interval == 0:
            logs = trainer.log()
            logs["env"] = aggregate_episode_logs(pending_env)
            pending_env.clear()
            entry = {
                "epoch": epoch,
                "agent_steps": trainer.global_step,
                "sps": logs["SPS"],
                "env": logs["env"],
                "loss": logs["loss"],
            }
            history.append(entry)
            candidate_score = score_episode_logs(logs["env"])
            if candidate_score > best_score:
                best_score = candidate_score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in trainer.policy.state_dict().items()
                }
            print(
                f"epoch={epoch}/{epochs} steps={trainer.global_step} "
                f"sps={logs['SPS']:.0f} env={logs['env']}",
                flush=True,
            )
    if best_state is not None:
        trainer.policy.load_state_dict(best_state)
    elapsed = perf_counter() - started
    return trainer, history, elapsed, {"epoch": best_epoch, "score": best_score}


def aggregate_episode_logs(logs: list[dict]) -> dict:
    episodes = sum(item.get("n", 0.0) for item in logs)
    if episodes == 0:
        return {}
    keys = set().union(*(item.keys() for item in logs)) - {"n"}
    result = {
        key: sum(item.get(key, 0.0) * item.get("n", 0.0) for item in logs) / episodes
        for key in keys
    }
    result["n"] = episodes
    return result


def score_episode_logs(logs: dict) -> float:
    if logs.get("n", 0.0) < 32.0:
        return float("-inf")
    return (
        3.0 * logs.get("success_rate", 0.0)
        - 2.0 * logs.get("collision_rate", 0.0)
        + 0.01 * logs.get("episode_return", 0.0)
    )


@torch.no_grad()
def evaluate(
    policy,
    args: dict,
    torch_pufferl,
    steps: int,
    vision_mode: str,
    obstacle_probability: float,
) -> dict:
    eval_args = deepcopy(args)
    eval_args["env"]["obstacle_probability"] = obstacle_probability
    vec = torch_pufferl._C.create_vec(eval_args, torch_pufferl._C.gpu)
    obs_dtype = torch.float32 if vec.obs_dtype == "FloatTensor" else torch.uint8
    observations = torch_pufferl._cpu_tensor(
        vec.obs_ptr,
        (vec.total_agents, vec.obs_size),
        obs_dtype,
    )
    terminals = torch_pufferl._cpu_tensor(
        vec.terminals_ptr,
        (vec.total_agents,),
        torch.float32,
    )
    vec.reset()
    state = policy.initial_state(vec.total_agents, device="cpu")
    vision_dim = VISION_CHANNELS * VISION_PIXELS
    for _ in range(steps):
        policy_obs = observations
        if vision_mode == "masked":
            policy_obs = observations.clone()
            policy_obs[:, :vision_dim] = 0.0
        distribution, _, state = policy.forward_eval(policy_obs, state)
        actions = distribution.mean.clamp(-1.0, 1.0).contiguous()
        vec.cpu_step(actions.data_ptr())
        alive = (1.0 - terminals).view(1, -1, 1)
        state = tuple(item * alive for item in state)
    metrics = dict(vec.log())
    vec.close()
    return {key: float(value) for key, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and causally evaluate the native Puffer visual-navigation policy."
    )
    parser.add_argument("--puffer-root", type=Path, default=DEFAULT_PUFFER_ROOT)
    parser.add_argument("--env-name", default="flightrl_visual_navigation_smoke")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "puffer_visual")
    parser.add_argument("--total-timesteps", type=int, default=262_144)
    parser.add_argument("--eval-steps", type=int, default=2_080)
    parser.add_argument("--agents", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=726)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--entropy-coef", type=float, default=0.0001)
    parser.add_argument("--action-logstd", type=float, default=-1.5)
    parser.add_argument("--reset-action-head", action="store_true")
    parser.add_argument("--obstacle-probability", type=float, default=0.75)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--log-interval", type=int, default=16)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    settings = Puffer4ExportSettings(
        env_name=args.env_name,
        total_agents=args.agents,
        num_buffers=1,
        num_threads=args.threads,
        policy_hidden_size=args.hidden_size,
        policy_num_layers=1,
        train_seed=args.seed,
    )
    result = export_visual_puffer4_assets(args.puffer_root, settings)
    if not args.skip_build:
        build_puffer_environment(args.puffer_root, result.env_name)
    puffer_args, torch_pufferl = load_puffer(args.puffer_root, result.env_name)
    puffer_args["train"]["learning_rate"] = args.learning_rate
    puffer_args["train"]["ent_coef"] = args.entropy_coef
    puffer_args["env"]["obstacle_probability"] = args.obstacle_probability
    if args.init_checkpoint:
        puffer_args["load_model_path"] = str(args.init_checkpoint.expanduser().resolve())
    trainer, history, elapsed, selection = train(
        puffer_args,
        torch_pufferl,
        args.total_timesteps,
        args.log_interval,
        args.action_logstd,
        args.reset_action_head,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"{args.env_name}_{args.total_timesteps}.bin"
    trainer.save_weights(checkpoint)
    obstacle_full = evaluate(
        trainer.policy, puffer_args, torch_pufferl, args.eval_steps, "full", 1.0
    )
    obstacle_masked = evaluate(
        trainer.policy, puffer_args, torch_pufferl, args.eval_steps, "masked", 1.0
    )
    clear_full = evaluate(
        trainer.policy, puffer_args, torch_pufferl, args.eval_steps, "full", 0.0
    )
    trainer.close()
    report = {
        "checkpoint": str(checkpoint),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "trainer": "PufferLib PuffeRL PPO with MinGRU",
        "learning_rate": args.learning_rate,
        "entropy_coef": args.entropy_coef,
        "initial_action_logstd": args.action_logstd,
        "reset_action_head": args.reset_action_head,
        "training_obstacle_probability": args.obstacle_probability,
        "policy_parameters": trainer.model_size,
        "observation": "3x48x64 gray4 appearance/delta/motion plus 6-value goal intent",
        "action": "learned residual over body vx/vy, world vz, and yaw-rate waypoint tracking",
        "total_timesteps": args.total_timesteps,
        "elapsed_s": elapsed,
        "history": history,
        "training_selection": selection,
        "evaluation": {
            "obstacle_full_vision": obstacle_full,
            "obstacle_masked_vision": obstacle_masked,
            "clear_full_vision": clear_full,
        },
        "camera_is_causal": obstacle_full.get("success_rate", 0.0)
        > obstacle_masked.get("success_rate", 0.0),
        "deployment_status": "simulation checkpoint only; not approved for live flight",
    }
    report_path = checkpoint.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"checkpoint={checkpoint}")
    print(f"report={report_path}")
    print(f"evaluation={report['evaluation']}")


if __name__ == "__main__":
    main()
