from __future__ import annotations


def require_reset_safe_fixed_door_ppo(total_timesteps: int) -> None:
    if total_timesteps <= 0:
        return
    raise RuntimeError(
        "generic fixed-door PuffeRL PPO is disabled because recurrent state is "
        "not masked at episode terminals; use total_timesteps=0 for reset-aware "
        "BC/DAgger until a local terminal-masked rollout and training path is "
        "verified"
    )


def fixed_door_gate(full: dict[str, float], masked: dict[str, float]) -> dict:
    checks = {
        "completion": full.get("success_rate", 0.0) >= 0.80,
        "collision": full.get("collision_rate", 1.0) <= 0.03,
        "outside_fov_completion": (
            full.get("outside_fov_success_rate", 0.0) >= 0.70
        ),
        "camera_mask": masked.get("success_rate", 1.0) <= 0.05,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
    }


def fixed_door_teacher_gate(metrics: dict[str, float]) -> dict:
    checks = {
        "completion": metrics.get("success_rate", 0.0) >= 0.93,
        "collision": metrics.get("collision_rate", 1.0) <= 0.02,
        "outside_fov_completion": (
            metrics.get("outside_fov_success_rate", 0.0) >= 0.90
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
    }
