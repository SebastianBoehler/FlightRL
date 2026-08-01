from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from math import isfinite
from pathlib import Path

from flightrl.evidence_scope import require_file_identity
from flightrl.puffer4_config import Puffer4ExportSettings, render_puffer4_ini
from flightrl.puffer4_edge_evaluation_gate import (
    EDGE_EVALUATION_SCHEMA,
    EDGE_EVALUATION_PROFILES,
    EDGE_STUDENT_GATE_THRESHOLDS,
    edge_student_gate,
)
from flightrl.puffer4_edge_evaluation_metrics import (
    require_evaluation_metric_consistency,
)
from flightrl.puffer4_edge_evidence import (
    ROOT,
    require_source_identities,
)
from flightrl.puffer4_edge_native_build import (
    require_current_edge_native_build_fingerprint,
)
from flightrl.puffer4_edge_student_sections import build_edge_student_sections


EDGE_EVALUATION_STEPS = 6000
EDGE_EVALUATION_AGENTS = 128
_AUTHORITY = {
    "authority": "none",
    "deployment_authority": False,
    "hardware_approved": False,
    "controls_drone": False,
}
_FIELDS = {
    "schema", "status", "scope", "checkpoint_identity",
    "policy_contract_sha256", "evaluated_target_ids", "profiles", "gate",
    "native_build_fingerprint", "environment_config_identity",
    "source_identity", *_AUTHORITY,
}
_PROFILE_FIELDS = {
    "metrics", "gate", "seed", "appearance_seed", "profile", "steps", "agents",
}
_NATIVE_METRICS = {
    "score", "episode_return", "episode_length", "success_rate",
    "collision_rate", "outside_fov_success_fraction",
    "outside_fov_episode_fraction", "outside_fov_observed_fraction",
    "observed_episode_fraction", "door_visible_fraction", "n",
    "low_light_episode_fraction", "low_light_success_fraction",
    "obstacle_episode_fraction", "obstacle_success_fraction",
    "scene_group_schema_version",
    *(
        f"{group}_{index}_{kind}_fraction"
        for group in ("layout_family", "door_face")
        for index in range(1, 4)
        for kind in ("episode", "success")
    ),
}
_DERIVED_METRICS = {
    "outside_fov_success_rate", "outside_fov_episodes", "episodes",
    "collision_rate_upper_95",
    "action_rmse", "door_action_rmse", "reset_action_rmse",
    "reset_door_action_rmse", "reset_samples", "lateral_action_abs_mean",
    "vertical_action_abs_mean", "lateral_action_abs_max",
    "vertical_action_abs_max", "action_saturation_fraction",
    "grounding_visibility_precision", "grounding_visibility_recall",
    "grounding_visible_box_mae", "grounding_visible_samples",
    "grounding_absent_samples", "hidden_min", "hidden_max",
}
_METRICS = _NATIVE_METRICS | _DERIVED_METRICS
_UNIT_INTERVAL = {
    name for name in _METRICS if name.endswith(("_rate", "_fraction"))
} | {
    "grounding_visibility_precision", "grounding_visibility_recall",
    "lateral_action_abs_mean", "vertical_action_abs_mean",
    "lateral_action_abs_max", "vertical_action_abs_max",
    "action_saturation_fraction",
}
_SOURCES = {
    "script": ROOT / "scripts/evaluate_puffer_edge_student.py",
    "artifact_paths": ROOT / "src/flightrl/artifact_paths.py",
    "evaluator": ROOT / "src/flightrl/puffer4_edge_evaluation.py",
    "gate": ROOT / "src/flightrl/puffer4_edge_evaluation_gate.py",
    "counts": ROOT / "src/flightrl/puffer4_edge_evaluation_counts.py",
    "metrics": ROOT / "src/flightrl/puffer4_edge_evaluation_metrics.py",
    "exporter": ROOT / "src/flightrl/puffer4_edge_student_export.py",
    "sections": ROOT / "src/flightrl/puffer4_edge_student_sections.py",
    "door_sections": ROOT / "src/flightrl/puffer4_door_sections.py",
    "config": ROOT / "src/flightrl/puffer4_config.py",
    "native_identity": ROOT / "src/flightrl/puffer4_edge_native_build.py",
}


def require_edge_evaluation_evidence(
    path: str | Path,
    *,
    checkpoint_identity: Mapping[str, str],
    trained_target_ids: Sequence[int],
    hidden_size: int,
    policy_contract_sha256: str,
    native_build_fingerprint: Mapping,
) -> None:
    report = _read_report(path)
    if set(report) != _FIELDS:
        raise ValueError("offline shadow evaluation fields are incompatible")
    if (
        report["schema"] != EDGE_EVALUATION_SCHEMA
        or report["status"] != "complete"
        or report["scope"] != "desktop_simulation_held_out"
    ):
        raise ValueError("offline shadow evaluation scope or completion is invalid")
    if (
        report["checkpoint_identity"] != checkpoint_identity
        or report["policy_contract_sha256"] != policy_contract_sha256
        or report["evaluated_target_ids"] != list(trained_target_ids)
    ):
        raise ValueError("offline shadow evaluation does not match its checkpoint")
    _require_authority(report)
    require_source_identities(
        report["source_identity"], _SOURCES, "edge evaluation"
    )
    fingerprint = require_current_edge_native_build_fingerprint(
        report["native_build_fingerprint"], expected=native_build_fingerprint
    )
    _require_environment_config(
        report["environment_config_identity"],
        fingerprint,
        hidden_size,
    )
    if report["gate"] != {"passed": True, "failures": []}:
        raise ValueError("offline shadow requires a passing held-out evaluation")
    profiles = report["profiles"]
    expected_names = [item[0] for item in EDGE_EVALUATION_PROFILES]
    if not isinstance(profiles, Mapping) or set(profiles) != set(expected_names):
        raise ValueError("offline shadow evaluation profile set is incomplete")
    failures = []
    for name, seed, appearance_seed, configuration in EDGE_EVALUATION_PROFILES:
        passed = _require_profile(
            name,
            profiles[name],
            seed,
            appearance_seed,
            configuration,
        )
        if not passed:
            failures.append(name)
    if report["gate"] != {"passed": not failures, "failures": failures}:
        raise ValueError("offline shadow evaluation aggregate gate is inconsistent")


def _require_profile(name, value, seed, appearance_seed, configuration) -> bool:
    if not isinstance(value, Mapping) or set(value) != _PROFILE_FIELDS:
        raise ValueError(f"edge evaluation {name} profile fields are invalid")
    if value["agents"] != EDGE_EVALUATION_AGENTS:
        raise ValueError(f"edge evaluation {name} agents are not canonical")
    if value["steps"] != EDGE_EVALUATION_STEPS:
        raise ValueError(f"edge evaluation {name} steps are not canonical")
    if value["seed"] != seed or value["appearance_seed"] != appearance_seed:
        raise ValueError(f"edge evaluation {name} seed is not canonical")
    if value["profile"] != configuration:
        raise ValueError(f"edge evaluation {name} profile configuration is invalid")
    metrics = _require_metrics(value["metrics"])
    require_evaluation_metric_consistency(
        metrics,
        configuration=configuration,
        steps=EDGE_EVALUATION_STEPS,
        agents=EDGE_EVALUATION_AGENTS,
    )
    derived = edge_student_gate(metrics, profile=configuration)
    if value["gate"] != derived:
        raise ValueError(f"edge evaluation {name} gate does not match its metrics")
    return derived["passed"] is True


def _require_metrics(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != _METRICS:
        raise ValueError("edge evaluation metric fields are incompatible")
    if any(not _finite_number(item) for item in value.values()):
        raise ValueError("edge evaluation requires finite numeric metrics")
    metrics = {name: float(item) for name, item in value.items()}
    if any(not 0.0 <= metrics[name] <= 1.0 for name in _UNIT_INTERVAL):
        raise ValueError("edge evaluation rate or fraction is outside [0, 1]")
    nonnegative = _DERIVED_METRICS - {"hidden_min", "hidden_max"}
    if any(metrics[name] < 0.0 for name in nonnegative):
        raise ValueError("edge evaluation derived metric is negative")
    if not set(EDGE_STUDENT_GATE_THRESHOLDS) <= set(metrics):
        raise ValueError("edge evaluation gate metrics are incomplete")
    return metrics


def _read_report(path: str | Path) -> Mapping:
    try:
        report = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("offline shadow evaluation report is unreadable") from exc
    if not isinstance(report, Mapping):
        raise ValueError("offline shadow evaluation report must be a mapping")
    return report


def _require_environment_config(value, fingerprint: Mapping, hidden_size: int) -> None:
    extension = Path(fingerprint["extension"]["path"])
    root = extension.parent.parent
    config_path = root / "config" / f"{fingerprint['env_name']}.ini"
    try:
        require_file_identity(value, config_path, label="edge evaluation environment config")
        actual = config_path.read_text()
    except OSError as exc:
        raise ValueError("edge evaluation environment config is unavailable") from exc
    settings = Puffer4ExportSettings(
        env_name=fingerprint["env_name"],
        total_agents=EDGE_EVALUATION_AGENTS,
        num_buffers=1,
        num_threads=8,
        policy_hidden_size=hidden_size,
        train_seed=17,
    )
    expected = render_puffer4_ini(build_edge_student_sections(settings))
    if actual != expected:
        raise ValueError("edge evaluation environment config is not canonical")


def _require_authority(report: Mapping) -> None:
    if any(report.get(name) != expected for name, expected in _AUTHORITY.items()):
        raise ValueError("offline shadow evaluation must be explicitly non-authoritative")


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(float(value))
    )
