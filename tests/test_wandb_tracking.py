from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

from flightrl.tracking import add_wandb_args, args_config, init_wandb, load_wandb_env, log_artifacts, log_metrics


def test_wandb_helper_uses_optional_module(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeArtifact:
        def __init__(self, name, type):
            calls.append(("artifact", name, type))

        def add_file(self, path):
            calls.append(("file", path))

    class FakeRun:
        def log(self, metrics, step=None):
            calls.append(("log", metrics, step))

        def log_artifact(self, artifact):
            calls.append(("log_artifact", artifact.__class__.__name__))

    def fake_init(**kwargs):
        calls.append(("init", kwargs))
        return FakeRun()

    parser = argparse.ArgumentParser()
    add_wandb_args(parser)
    args = parser.parse_args(["--wandb", "--wandb-mode", "offline", "--wandb-tags", "a,b"])
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init, Artifact=FakeArtifact))

    run = init_wandb(args, {"epochs": 2})
    output = tmp_path / "checkpoint.pt"
    output.write_text("ok")
    log_metrics(run, {"loss": 1.0}, step=2)
    log_artifacts(run, name="candidate", paths=[output], artifact_type="model")

    assert calls[0][0] == "init"
    assert calls[0][1]["project"] == "FlightRL"
    assert calls[0][1]["tags"] == ["a", "b"]
    assert ("log", {"loss": 1.0}, 2) in calls


def test_wandb_is_enabled_by_default_and_can_be_disabled() -> None:
    parser = argparse.ArgumentParser()
    add_wandb_args(parser)

    assert parser.parse_args([]).wandb is True
    assert parser.parse_args(["--no-wandb"]).wandb is False


def test_wandb_mode_defaults_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    parser = argparse.ArgumentParser()
    add_wandb_args(parser)

    assert parser.parse_args([]).wandb_mode == "disabled"


def test_load_wandb_env_file(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / "wandb.env"
    env_file.write_text("export WANDB_API_KEY=secret\nWANDB_PROJECT=FlightRL\n")
    monkeypatch.setenv("WANDB_API_KEY", "stale")
    monkeypatch.delenv("WANDB_PROJECT", raising=False)

    load_wandb_env(env_file)

    assert os.environ["WANDB_API_KEY"] == "secret"
    assert os.environ["WANDB_PROJECT"] == "FlightRL"


def test_args_config_excludes_wandb_fields() -> None:
    args = SimpleNamespace(epochs=2, checkpoint="x.pt", wandb_project="ignored", inputs=["a"])

    assert args_config(args) == {"epochs": 2, "checkpoint": "x.pt", "inputs": ["a"]}
