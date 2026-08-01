from __future__ import annotations


def semantic_puffer_config(
    total_timesteps: int,
    total_agents: int,
    horizon: int,
    learning_rate: float,
) -> dict:
    return {
        "world_size": 1,
        "train": {
            "total_timesteps": total_timesteps,
            "learning_rate": learning_rate,
            "anneal_lr": True,
            "min_lr_ratio": 0.1,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "replay_ratio": 2.0,
            "clip_coef": 0.2,
            "vf_coef": 0.5,
            "vf_clip_coef": 0.2,
            "max_grad_norm": 1.0,
            "ent_coef": 0.0003,
            "beta1": 0.9,
            "eps": 1e-8,
            "minibatch_size": total_agents * horizon,
            "horizon": horizon,
            "vtrace_rho_clip": 1.0,
            "vtrace_c_clip": 1.0,
            "prio_alpha": 0.8,
            "prio_beta0": 0.2,
        },
    }
