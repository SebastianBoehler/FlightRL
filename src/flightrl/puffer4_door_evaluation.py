from __future__ import annotations

from flightrl.puffer4_door_training import (
    evaluate_door_policy,
    fixed_door_gate,
)


def screen_score(full: dict) -> tuple[float, float, float]:
    return (
        full.get("success_rate", 0.0) - 3.0 * full.get("collision_rate", 1.0),
        full.get("outside_fov_success_rate", 0.0),
        full.get("door_visible_fraction", 0.0),
    )


def evaluate_door_candidates(
    trainer,
    states: dict,
    args: dict,
    torch_pufferl,
    *,
    screen_steps: int,
    eval_steps: int,
    seed: int,
    agents: int,
) -> tuple[str, dict, dict]:
    screens = {}
    for name, state in states.items():
        trainer.policy.load_state_dict(state)
        screens[name] = evaluate_door_policy(
            trainer.policy,
            args,
            torch_pufferl,
            steps=screen_steps,
            seed=seed,
            camera_mask=False,
            agents=agents,
        )
    selected = max(screens, key=lambda name: screen_score(screens[name]))
    trainer.policy.load_state_dict(states[selected])
    screen = screens[selected]
    promising = (
        screen.get("success_rate", 0.0) >= 0.50
        and screen.get("collision_rate", 1.0) <= 0.10
    )
    if not promising:
        masked = {
            "status": "skipped_candidate_failed_full_camera_screen",
            "success_rate": 1.0,
        }
        return selected, screens, {
            "full_camera": screen,
            "masked_camera": masked,
            "gate": fixed_door_gate(screen, masked),
            "evaluation_mode": "failed_fast_after_full_camera_screen",
        }
    full = evaluate_door_policy(
        trainer.policy,
        args,
        torch_pufferl,
        steps=eval_steps,
        seed=seed,
        camera_mask=False,
        agents=agents,
    )
    masked = evaluate_door_policy(
        trainer.policy,
        args,
        torch_pufferl,
        steps=eval_steps,
        seed=seed,
        camera_mask=True,
        agents=agents,
    )
    return selected, screens, {
        "full_camera": full,
        "masked_camera": masked,
        "gate": fixed_door_gate(full, masked),
        "evaluation_mode": "full_camera_and_masked_camera",
    }
