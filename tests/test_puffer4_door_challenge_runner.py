from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from flightrl.puffer4_door_challenge_runner import (
    run_door_challenge_evaluation,
)
from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
)
from flightrl.puffer4_door_eval_provenance import (
    begin_fixed_door_evaluation_provenance,
    fixed_door_generated_paths,
)

def test_challenge_runner_module_exists() -> None:
    assert importlib.util.find_spec(
        "flightrl.puffer4_door_challenge_runner"
    ) is not None


class _Vec:
    total_agents = 2
    obs_size = 4

    def __init__(self) -> None:
        self.observations = torch.zeros((2, 4))
        self.terminals = torch.zeros(2)
        self.obs_ptr = self.observations.data_ptr()
        self.terminals_ptr = self.terminals.data_ptr()

    def reset(self) -> None:
        self.observations.zero_()
        self.terminals.zero_()

    def cpu_step(self, _actions_ptr: int) -> None:
        self.observations.add_(1.0)
        self.terminals.zero_()

    def log(self) -> dict[str, float]:
        return {
            "n": 10.0,
            "success_rate": 0.6,
            "collision_rate": 0.1,
            "outside_fov_episode_fraction": 0.5,
            "outside_fov_success_fraction": 0.3,
        }

    def close(self) -> None:
        return None


class _Puffer:
    def __init__(self) -> None:
        self.vec: _Vec | None = None
        self.created_args: dict | None = None
        self._C = SimpleNamespace(create_vec=self._create_vec, gpu=0)

    def _create_vec(self, args: dict, _gpu: int) -> _Vec:
        self.created_args = args
        self.vec = _Vec()
        return self.vec

    def _cpu_tensor(
        self,
        pointer: int,
        _shape: tuple[int, ...],
        _dtype: torch.dtype,
    ) -> torch.Tensor:
        assert self.vec is not None
        return {
            self.vec.obs_ptr: self.vec.observations,
            self.vec.terminals_ptr: self.vec.terminals,
        }[pointer]


class _Policy:
    def initial_state(
        self,
        batch_size: int,
        device: str,
    ) -> tuple[torch.Tensor]:
        return (torch.zeros((1, batch_size, 1), device=device),)

    def forward_eval(
        self,
        observations: torch.Tensor,
        state: tuple[torch.Tensor],
    ) -> tuple[SimpleNamespace, torch.Tensor, tuple[torch.Tensor]]:
        means = torch.zeros((observations.shape[0], 2))
        values = torch.zeros((observations.shape[0], 1))
        return SimpleNamespace(mean=means), values, state


def _trained_identity(checkpoint: Path) -> dict:
    return {
        "checkpoint": {"path": str(checkpoint), "sha256": "checkpoint-sha"},
        "action_contract": {"sha256": "action-sha"},
        "policy_contract": {"sha256": "policy-sha"},
        "environment": {"name": "flightrl_fixed_door_d1"},
        "train_seed": 11,
    }


def _write_control(
    path: Path,
    trained_identity: dict,
    native_build_fingerprint: dict,
) -> None:
    path.write_text(
        json.dumps(
            {
                "evaluation_schema": "flightrl.fixed_door.promotion.v3",
                "trained_identity": trained_identity,
                "evaluation_identity": {
                    "kind": "fixed_door_promotion",
                    "schema_version": 1,
                    "report": {"path": str(path.resolve())},
                    "environment": {
                        "name": "flightrl_fixed_door_d1",
                        "seed": 31,
                        "steps_per_condition": 2,
                        "agents": 2,
                    },
                    "native_build_fingerprint": native_build_fingerprint,
                    "action_contract_sha256": "action-sha",
                    "policy_contract_sha256": "policy-sha",
                    "procedural_stream_contract": {"contract_id": "stream"},
                    "evidence_age_runtime_contract": (
                        FIXED_DOOR_EVIDENCE_AGE_CONTRACT.to_report()
                    ),
                },
                "full_camera": {
                    "status": "complete",
                    "success_rate": 0.8,
                    "outside_fov_success_rate": 0.7,
                    "collision_rate": 0.01,
                    "finite_outputs": {"passed": True},
                },
            }
        )
    )


def test_runner_executes_one_resolved_challenge_and_writes_exclusively(
    tmp_path: Path,
) -> None:
    checkpoint = (tmp_path / "door.bin").resolve()
    checkpoint.write_bytes(b"checkpoint")
    trained = _trained_identity(checkpoint)
    native = {"extension_sha256": "native-sha"}
    control = tmp_path / "door.promotion-evaluation.json"
    _write_control(control, trained, native)
    output = tmp_path / "dark.challenge-evaluation.json"
    lineage_report = tmp_path / "door.report.json"
    lineage_report.write_text('{"lineage": true}')
    root = Path(__file__).resolve().parents[1]
    provenance_capture = begin_fixed_door_evaluation_provenance(
        command=("python", "evaluate.py", "--challenge", "fixed-dark"),
        flightrl_root=root,
        entrypoint=root / "scripts/evaluate_puffer_fixed_door_checkpoint.py",
    )
    puffer_root = tmp_path / "Puffer"
    for path in fixed_door_generated_paths(puffer_root, "door_env"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
    bundle = SimpleNamespace(
        checkpoint_path=checkpoint,
        report_path=lineage_report,
        env_name="door_env",
        trained_identity=lambda: trained,
        lineage=lambda: {"report": {"path": "/tmp/train.json", "sha256": "x"}},
    )
    puffer = _Puffer()
    baseline_env = {
        "seed": 1,
        "camera_mask": 0.0,
        "control_dt": 1.0 / 65.0,
        "maximum_evidence_age_s": 1.0,
        "camera_mean_min": 18.0,
        "camera_mean_max": 110.0,
        "camera_randomization": 0.0,
        "obstacle_probability": 0.0,
        "layout_diversity": 1.0,
        "room_x_min": -2.0,
        "room_x_max": 2.0,
        "room_y_min": -2.0,
        "room_y_max": 2.0,
    }

    report, written = run_door_challenge_evaluation(
        bundle=bundle,
        policy=_Policy(),
        puffer_args={"env": baseline_env, "vec": {}},
        torch_pufferl=puffer,
        challenge="fixed-dark",
        control_report=control,
        output=output,
        native_build_fingerprint=native,
        stream_contract={"contract_id": "stream"},
        provenance_capture=provenance_capture,
        puffer_root=puffer_root,
        steps=2,
        seed=31,
        agents=2,
    )

    assert written == output.resolve()
    assert json.loads(output.read_text()) == report
    assert report["challenge"]["metrics"]["success_rate"] == 0.6
    assert report["evaluation_provenance"]["command"][-2:] == [
        "--challenge",
        "fixed-dark",
    ]
    assert report["challenge"]["resolved_single_variable"][
        "environment_overrides"
    ] == {"camera_mean_min": 20.0, "camera_mean_max": 20.0}
    assert puffer.created_args is not None
    assert puffer.created_args["env"]["camera_mean_min"] == 20.0
    assert puffer.created_args["env"]["camera_mean_max"] == 20.0
    assert puffer.created_args["env"]["camera_randomization"] == 0.0
    assert baseline_env["camera_mean_min"] == 18.0
