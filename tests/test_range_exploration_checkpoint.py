from __future__ import annotations

from pathlib import Path

import pytest
import torch

from flightrl.exploration.range_checkpoint import (
    load_range_checkpoint,
    range_training_contract,
    require_shadow_eligible_range_checkpoint,
    save_range_checkpoint,
)
from flightrl.exploration.range_evaluation import evaluate_range_candidate
from flightrl.exploration.range_policy import RangeExplorationActorCritic


def _training(seed: int, updates: int) -> dict[str, int | float]:
    return range_training_contract(
        seed=seed,
        updates=updates,
        num_envs=1,
        rollout_horizon=1,
        learning_rate=3e-4,
        action_std=0.25,
        frontier_aux_coef=0.0,
        shield_aux_coef=0.10,
        general_turn_commitment_coef=0.0,
        obstacle_turn_commitment_coef=0.10,
    )


def _candidate(tmp_path: Path) -> Path:
    torch.manual_seed(603)
    model = RangeExplorationActorCritic(hidden_size=64)
    report = evaluate_range_candidate(model, seeds=(603,), horizon=5)
    output = tmp_path / "candidate.pt"
    save_range_checkpoint(
        output,
        model,
        report,
        training=_training(603, 0),
        source_revision="1ac9a0c1d63ab6e3781bf5cfd2c8873521d462fc",
    )
    return output


def test_failed_candidate_is_reloadable_but_not_shadow_eligible(tmp_path: Path) -> None:
    checkpoint = _candidate(tmp_path)

    model, report = load_range_checkpoint(checkpoint)

    assert model.parameter_count < 100_000
    assert report["simulation_gate_passed"] is False
    with pytest.raises(ValueError, match="simulation gate"):
        require_shadow_eligible_range_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(contract_id="wrong"), "contract"),
        (lambda value: value["authority"].update(flight=True), "authority"),
        (lambda value: value.update(state_sha256="0" * 64), "state"),
        (lambda value: value["evaluation"].update(simulation_gate_passed=True), "checks"),
        (
            lambda value: value["training"].update(obstacle_turn_commitment_coef=0.0),
            "training",
        ),
    ],
)
def test_checkpoint_rejects_forged_contract_state_or_authority(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    checkpoint = _candidate(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    mutation(payload)
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match=message):
        load_range_checkpoint(checkpoint)


def test_checkpoint_recomputes_claimed_checks_from_stored_metrics(tmp_path: Path) -> None:
    checkpoint = _candidate(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["evaluation"]["checks"] = {
        name: True for name in payload["evaluation"]["checks"]
    }
    payload["evaluation"]["simulation_gate_passed"] = True
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="metrics do not support"):
        load_range_checkpoint(checkpoint)


def test_checkpoint_rejects_short_envelope_even_if_metrics_pass(tmp_path: Path) -> None:
    checkpoint = _candidate(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    evaluation = payload["evaluation"]
    clean = evaluation["modes"]["clean"]
    clean.update(
        mean_final_objective=1.0,
        mean_objective_auc=1.0,
        mean_final_coverage=1.0,
        collision_rate=0.0,
        safety_terminal_rate=0.0,
        challenge_rate=1.0,
    )
    for name in ("range_masked", "map_masked"):
        evaluation["modes"][name]["mean_final_coverage"] = 0.0
    evaluation["modes"]["stress"].update(
        collision_rate=0.0,
        safety_terminal_rate=0.0,
    )
    for baseline in evaluation["baselines"].values():
        baseline.update(mean_final_objective=0.0, mean_objective_auc=0.0)
    evaluation["counterfactuals"] = {
        "mirrored_frontier_direction": True,
        "front_obstacle_response": True,
    }
    evaluation["obstacle_challenge"].update(
        challenge_rate=1.0,
        escape_rate=1.0,
        collision_rate=0.0,
        safety_terminal_rate=0.0,
    )
    evaluation["checks"] = {
        "beats_stationary_and_classical_coverage": True,
        "zero_clean_collisions": True,
        "zero_clean_safety_terminals": True,
        "dedicated_obstacle_challenge": True,
        "range_causal": True,
        "map_causal": True,
        "stress_collision_free": True,
        "front_obstacle_response": True,
    }
    evaluation["simulation_gate_passed"] = True
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="full evaluation envelope"):
        load_range_checkpoint(checkpoint)
