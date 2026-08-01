from __future__ import annotations

import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import re
import tempfile

import torch

from flightrl import puffer4_edge_replay_identity as replay_identity
from flightrl.evidence_scope import (
    file_identity,
    require_existing_file_identity,
)
from flightrl.puffer4_edge_checkpoint import (
    EdgeCheckpointMetadata,
    load_edge_checkpoint,
)
from flightrl.puffer4_edge_native_build import (
    require_matching_edge_native_build_fingerprints,
)
from flightrl.puffer4_edge_sequence import (
    EdgeSequenceDataset,
    load_edge_sequence_dataset,
    require_disjoint_edge_dataset_structures,
    require_edge_sequence_structure,
    require_matching_edge_dataset_environments,
)
from flightrl.puffer4_edge_training import (
    EDGE_TRAINING_REPORT_SCHEMA,
    apply_recurrent_resets,
)


EDGE_REPLAY_SCHEMA = "flightrl.edge_v3.offline_passive_replay.v2"
_AUTHORITY = {
    "authority": "none",
    "deployment_authority": False,
    "hardware_approved": False,
    "controls_drone": False,
}


@torch.no_grad()
def write_edge_passive_replay(
    *,
    checkpoint: str | Path,
    dataset: str | Path,
    output: str | Path,
) -> dict:
    identities, output_path = replay_identity.capture_inputs(
        checkpoint, dataset, output
    )
    actor, metadata = load_edge_checkpoint(identities["checkpoint_identity"]["path"])
    replay_identity.require_training_output_distinct(metadata, output_path)
    sequence = load_edge_sequence_dataset(identities["dataset_identity"]["path"])
    replay_identity.require_unchanged(identities, "while loading")
    _require_final_dataset(sequence, metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = actor.initial_state(sequence.shape[1])
    error_sum = 0.0
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=output_path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            _write_record(
                handle,
                _header(identities, metadata, sequence),
            )
            for step in range(sequence.shape[0]):
                record, state, squared_error = _replay_step(
                    actor,
                    sequence,
                    state,
                    step,
                )
                error_sum += squared_error
                _write_record(handle, record)
            summary = _summary(sequence, error_sum)
            _write_record(handle, summary)
            handle.flush()
            os.fsync(handle.fileno())
        replay_identity.require_unchanged(identities, "before atomic replace")
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return summary | {"output_identity": file_identity(output_path)}


@torch.no_grad()
def require_edge_passive_replay(
    path: str | Path,
    *,
    checkpoint_context: tuple[Path, object, EdgeCheckpointMetadata] | None = None,
) -> dict:
    replay = Path(path)
    try:
        with replay.open(encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("offline passive replay is unreadable") from exc
    if len(lines) < 3:
        raise ValueError("offline passive replay is incomplete")
    header, summary = lines[0], lines[-1]
    _require_header(header)
    checkpoint_identity = require_existing_file_identity(
        header["checkpoint_identity"],
        label="offline replay checkpoint",
    )
    dataset_identity = require_existing_file_identity(
        header["dataset_identity"],
        label="offline replay dataset",
    )
    if checkpoint_context is None:
        actor, metadata = load_edge_checkpoint(checkpoint_identity["path"])
    else:
        checkpoint_path, actor, metadata = checkpoint_context
        if checkpoint_identity != file_identity(checkpoint_path):
            raise ValueError("offline passive replay checkpoint context does not match")
    sequence = load_edge_sequence_dataset(dataset_identity["path"])
    _require_final_dataset(sequence, metadata)
    if (
        header["policy_contract_sha256"] != metadata.policy_contract_sha256
        or header["trained_target_ids"] != list(metadata.trained_target_ids)
        or header["native_build_fingerprint"] != metadata.native_build_fingerprint
        or header["steps"] != sequence.shape[0]
        or header["agents"] != sequence.shape[1]
    ):
        raise ValueError("offline passive replay header does not match its artifacts")
    if len(lines) != sequence.shape[0] + 2:
        raise ValueError("offline passive replay record count is inconsistent")
    state = actor.initial_state(sequence.shape[1])
    error_sum = 0.0
    for step, actual in enumerate(lines[1:-1]):
        expected, state, squared_error = _replay_step(
            actor,
            sequence,
            state,
            step,
        )
        error_sum += squared_error
        if actual != expected:
            raise ValueError("offline passive replay step does not reproduce")
    _require_authority(summary)
    if summary != _summary(sequence, error_sum):
        raise ValueError("offline passive replay summary does not reproduce")
    return {
        "header": header,
        "summary": summary,
        "dataset_metadata": sequence.metadata,
    }


def _require_final_dataset(
    sequence: EdgeSequenceDataset,
    checkpoint: EdgeCheckpointMetadata,
) -> None:
    require_edge_sequence_structure(sequence)
    if sequence.metadata["split"] != "final":
        raise ValueError("offline passive replay requires a final held-out dataset")
    if sequence.metadata["policy_contract_sha256"] != checkpoint.policy_contract_sha256:
        raise ValueError("offline replay dataset contract does not match checkpoint")
    observed = sorted(int(value) for value in set(sequence.target_ids.flat))
    if any(value not in checkpoint.trained_target_ids for value in observed):
        raise ValueError("offline replay target is not covered by the checkpoint")
    training = _training_datasets(checkpoint)
    require_disjoint_edge_dataset_structures(training[0], training[1], sequence)
    require_matching_edge_native_build_fingerprints(
        checkpoint.native_build_fingerprint,
        *(dataset.metadata["native_build_fingerprint"] for dataset in (*training, sequence)),
    )
    require_matching_edge_dataset_environments(*training, sequence)


def _training_datasets(
    checkpoint: EdgeCheckpointMetadata,
) -> tuple[EdgeSequenceDataset, EdgeSequenceDataset]:
    report_path = require_existing_file_identity(
        checkpoint.training_identity,
        label="edge replay training report",
    )["path"]
    report = json.loads(Path(report_path).read_text())
    identities = report.get("datasets")
    if (
        report.get("schema") != EDGE_TRAINING_REPORT_SCHEMA
        or not isinstance(identities, dict)
        or set(identities) != {"train", "selection"}
    ):
        raise ValueError("offline replay training dataset identities are invalid")
    datasets = []
    for split in ("train", "selection"):
        identity = require_existing_file_identity(
            identities[split],
            label=f"edge replay {split} dataset",
        )
        datasets.append(
            load_edge_sequence_dataset(
                identity["path"],
                verify_execution_trace=False,
            )
        )
    if file_identity(identities["selection"]["path"]) != checkpoint.dataset_identity:
        raise ValueError("checkpoint selection dataset identity is inconsistent")
    return datasets[0], datasets[1]


def _header(identities, metadata, sequence) -> dict:
    return {
        "schema": EDGE_REPLAY_SCHEMA,
        "record": "header",
        "mode": "offline_passive_shadow",
        **identities,
        "policy_contract_sha256": metadata.policy_contract_sha256,
        "trained_target_ids": list(metadata.trained_target_ids),
        "native_build_fingerprint": metadata.native_build_fingerprint,
        "steps": sequence.shape[0],
        "agents": sequence.shape[1],
        **_AUTHORITY,
    }


def _require_header(header: object) -> None:
    fields = {
        "schema", "record", "mode", "checkpoint_identity", "dataset_identity",
        "policy_contract_sha256", "trained_target_ids", "steps", "agents",
        "native_build_fingerprint",
        *_AUTHORITY,
    }
    if not isinstance(header, dict) or set(header) != fields:
        raise ValueError("offline passive replay header fields are invalid")
    if (
        header["schema"] != EDGE_REPLAY_SCHEMA
        or header["record"] != "header"
        or header["mode"] != "offline_passive_shadow"
    ):
        raise ValueError("offline passive replay header schema is invalid")
    _require_authority(header)


def _replay_step(actor, sequence, state, step):
    state = apply_recurrent_resets(state, sequence.resets[step])
    action, grounding, state = actor.forward_step(
        sequence.model_observation(step),
        state,
    )
    teacher = torch.from_numpy(sequence.teacher_actions[step])
    record = {
        "record": "step",
        "step": step,
        "reset_count": int(sequence.resets[step].sum()),
        "done_count": int(sequence.dones[step].sum()),
        "action_sha256": _tensor_sha256(action),
        "grounding_sha256": _tensor_sha256(grounding),
        "hidden_sha256": _tensor_sha256(state),
    }
    return record, state, float((action - teacher).square().sum())


def _summary(sequence, error_sum):
    values = sequence.shape[0] * sequence.shape[1] * 4
    rmse = (error_sum / values) ** 0.5
    if not isfinite(rmse):
        raise RuntimeError("offline passive replay RMSE is nonfinite")
    return {
        "record": "summary",
        "steps": sequence.shape[0],
        "agents": sequence.shape[1],
        "action_rmse": rmse,
        "complete": True,
        **_AUTHORITY,
    }


def _require_authority(record: dict) -> None:
    if any(record.get(key) != value for key, value in _AUTHORITY.items()):
        raise ValueError("offline passive replay cannot carry control authority")


def _write_record(handle, value: dict) -> None:
    handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _tensor_sha256(value: torch.Tensor) -> str:
    data = value.detach().cpu().contiguous().numpy().tobytes()
    digest = hashlib.sha256(data).hexdigest()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RuntimeError("offline passive replay tensor digest is invalid")
    return digest
