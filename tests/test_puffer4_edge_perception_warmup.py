from __future__ import annotations

import pytest
import torch

import flightrl.puffer4_edge_perception_warmup as warmup
import flightrl.puffer4_edge_training as training
from flightrl.puffer4_edge_policy import EdgeNavigationActor
from flightrl.puffer4_edge_training import EdgeTrainConfig
from flightrl.puffer4_edge_training import EdgeTrainingRejected
from flightrl.puffer4_edge_training_evidence import (
    require_perception_warmup_evidence,
)
from flightrl.puffer4_edge_training_state import edge_state_dict_sha256
from puffer4_edge_training_support import training_dataset


def _config(**overrides) -> EdgeTrainConfig:
    values = {
        "epochs": 1,
        "tbptt_steps": 2,
        "warmup_epochs": 2,
        "warmup_batch_size": 4,
        "seed": 17,
    }
    values.update(overrides)
    return EdgeTrainConfig(**values)


def test_warmup_orders_cover_every_flat_sample_once_and_are_deterministic() -> None:
    first = list(warmup.edge_perception_batch_orders(16, 4, 2, seed=17))
    second = list(warmup.edge_perception_batch_orders(16, 4, 2, seed=17))

    assert len(first) == 2
    assert first == second
    assert first[0] != first[1]
    for epoch in first:
        assert len(epoch) == 4
        flattened = [index for batch in epoch for index in batch]
        assert sorted(flattened) == list(range(16))
        assert len(flattened) == len(set(flattened))


def test_warmup_order_rejects_partial_or_oversized_batches() -> None:
    with pytest.raises(ValueError, match="divide evenly"):
        list(warmup.edge_perception_batch_orders(15, 4, 2, seed=17))
    with pytest.raises(ValueError, match="divide evenly"):
        list(warmup.edge_perception_batch_orders(4, 8, 2, seed=17))


def test_warmup_changes_only_perception_then_freezes_selected_state() -> None:
    actor = EdgeNavigationActor()
    control_before = edge_state_dict_sha256(
        warmup.edge_control_state_dict(actor)
    )

    evidence = warmup.warmup_edge_perception(
        actor,
        training_dataset("train", 11),
        training_dataset("selection", 21),
        _config(),
    )

    assert edge_state_dict_sha256(warmup.edge_control_state_dict(actor)) == (
        control_before
    )
    assert evidence["selected_epoch"] in {1, 2}
    assert evidence["selected_epoch"] == min(
        evidence["history"],
        key=lambda record: record["selection"]["grounding_loss"],
    )["epoch"]
    assert evidence["selected_state_sha256"] == (
        warmup.edge_perception_state_sha256(actor)
    )
    assert evidence["frozen_parameter_names"] == list(
        warmup.edge_perception_parameter_names(actor)
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in actor.named_parameters()
        if name in evidence["frozen_parameter_names"]
    )
    assert all(
        parameter.requires_grad
        for name, parameter in actor.named_parameters()
        if name not in evidence["frozen_parameter_names"]
    )


def test_warmup_reloads_minimum_selection_epoch_not_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = EdgeNavigationActor()
    captured: list[dict[str, torch.Tensor]] = []
    metrics = iter(
        (
            {"visibility_loss": 0.4, "box_loss": 0.1, "grounding_loss": 0.14},
            {"visibility_loss": 0.6, "box_loss": 0.1, "grounding_loss": 0.20},
        )
    )
    original_epoch = warmup._train_perception_epoch

    def capture_epoch(*args, **kwargs):
        original_epoch(*args, **kwargs)
        captured.append(
            {
                name: value.detach().clone()
                for name, value in warmup.edge_perception_state_dict(actor).items()
            }
        )

    monkeypatch.setattr(warmup, "_train_perception_epoch", capture_epoch)
    monkeypatch.setattr(
        warmup,
        "evaluate_edge_grounding",
        lambda *_args, **_kwargs: next(metrics),
    )

    evidence = warmup.warmup_edge_perception(
        actor,
        training_dataset("train", 11),
        training_dataset("selection", 21),
        _config(),
    )

    assert evidence["selected_epoch"] == 1
    assert all(
        torch.equal(actor.state_dict()[name], expected)
        for name, expected in captured[0].items()
    )
    assert any(
        not torch.equal(captured[0][name], captured[1][name])
        for name in captured[0]
    )


def test_warmup_evidence_records_exact_sampling_contract() -> None:
    actor = EdgeNavigationActor()
    evidence = warmup.warmup_edge_perception(
        actor,
        training_dataset("train", 11),
        training_dataset("selection", 21),
        _config(),
    )

    assert evidence["sampling"] == {
        "rng": "numpy.PCG64",
        "flattening": "step_major_agent_minor",
        "order": "full_permutation_without_replacement",
        "samples_per_epoch": 16,
        "batches_per_epoch": 4,
    }
    assert [record["epoch"] for record in evidence["history"]] == [1, 2]


def test_warmup_evidence_reproduces_and_rejects_forged_state_digest() -> None:
    actor = EdgeNavigationActor()
    train = training_dataset("train", 11)
    selection = training_dataset("selection", 21)
    config = _config()
    evidence = warmup.warmup_edge_perception(
        actor, train, selection, config
    )

    require_perception_warmup_evidence(
        evidence, config, actor, train, selection
    )
    forged = dict(evidence)
    forged["selected_state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="state digest"):
        require_perception_warmup_evidence(
            forged, config, actor, train, selection
        )


def test_trainer_uses_fresh_disjoint_optimizers_and_reports_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[set[int], float]] = []
    real_adamw = torch.optim.AdamW

    def tracked_adamw(parameters, *args, **kwargs):
        materialized = tuple(parameters)
        calls.append(
            ({id(parameter) for parameter in materialized}, float(kwargs["lr"]))
        )
        return real_adamw(materialized, *args, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", tracked_adamw)
    monkeypatch.setattr(
        training,
        "require_edge_training_coverage",
        lambda *_args: {"train": {}, "selection": {}},
    )
    try:
        _actor, report = training.train_edge_student(
            training_dataset("train", 11),
            training_dataset("selection", 21),
            _config(
                epochs=1,
                perception_learning_rate=2.0e-3,
                learning_rate=5.0e-4,
            ),
        )
    except EdgeTrainingRejected as exc:
        report = exc.report

    assert len(calls) == 2
    assert calls[0][0]
    assert calls[1][0]
    assert calls[0][0].isdisjoint(calls[1][0])
    assert calls[0][1] == pytest.approx(2.0e-3)
    assert calls[1][1] == pytest.approx(5.0e-4)
    assert report["schema"] == "flightrl.edge_v3.training_report.v5"
    assert set(report["perception_warmup"]) == warmup.WARMUP_REPORT_FIELDS


def test_trainer_checks_coverage_before_seed_actor_or_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        training,
        "require_edge_training_coverage",
        lambda *_args: (_ for _ in ()).throw(ValueError("coverage sentinel")),
    )
    monkeypatch.setattr(
        training.torch,
        "manual_seed",
        lambda *_args: (_ for _ in ()).throw(AssertionError("seeded too early")),
    )
    monkeypatch.setattr(
        training,
        "EdgeNavigationActor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("actor created too early")
        ),
    )

    with pytest.raises(ValueError, match="coverage sentinel"):
        training.train_edge_student(
            training_dataset("train", 11),
            training_dataset("selection", 21),
            _config(epochs=1),
        )
