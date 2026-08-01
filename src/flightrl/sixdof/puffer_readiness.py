from __future__ import annotations

from flightrl.evidence_values import exact_nonnegative_int, exact_true, failure_strings
from flightrl.puffer4_sixdof_sections import TASK_IDS


REQUIRED_CHECKS = {
    "files_present",
    "binding_tokens",
    "config_env_name",
    "config_total_agents",
    "config_num_buffers",
    "config_num_threads",
    "config_hidden_size",
    "room_bounds_present",
    "native_step_include_copied",
    "native_step_include_referenced",
    "physics_knobs_present",
    "sensor_profile_knobs_present",
    "range_observation_flag_present",
    "task_id_present",
    "reward_mode_present",
    "reset_profile_knobs_present",
}


def compact_puffer_export(report: dict | None) -> dict:
    if report is None:
        return {"present": False, "passed": False}
    if not isinstance(report, dict):
        raise ValueError("Puffer export report must be an object")
    checks = report.get("checks")
    raw_names = [
        check.get("name") for check in checks if isinstance(check, dict)
    ] if isinstance(checks, list) else []
    check_names = set(raw_names) if all(isinstance(name, str) for name in raw_names) else set()
    checks_passed = (
        isinstance(checks, list)
        and check_names == REQUIRED_CHECKS
        and len(checks) == len(check_names)
        and failure_strings(report.get("failures", [])) == []
        and all(
            isinstance(check, dict)
            and exact_true(check.get("passed"))
            and failure_strings(check.get("failures", [])) == []
            for check in checks
        )
    )
    env_name = report.get("env_name")
    config = report.get("config")
    files = report.get("files")
    passed = (
        exact_true(report.get("passed"))
        and checks_passed
        and isinstance(env_name, str)
        and env_name.startswith("flightrl_sixdof")
        and valid_config(config, env_name)
        and isinstance(files, dict)
        and bool(files)
        and all(valid_export_file(value) for value in files.values())
    )
    return {
        "present": True,
        "passed": passed,
        "env_name": env_name,
        "checks": checks if isinstance(checks, list) else [],
        "config": config if isinstance(config, dict) else {},
        "files": files if isinstance(files, dict) else {},
    }


def puffer_export_failures(
    report: dict,
    *,
    require: bool,
    candidate: dict | None = None,
) -> list[str]:
    if type(require) is not bool:
        raise ValueError("require_puffer_export must be a boolean")
    if not require:
        return []
    if not isinstance(report, dict) or not exact_true(report.get("present")):
        return ["puffer_export_missing"]
    if not exact_true(report.get("passed")):
        return ["puffer_export"]
    if not isinstance(candidate, dict) or not matches_candidate(report, candidate):
        return ["puffer_export_contract"]
    return []


def matches_candidate(report: dict, candidate: dict) -> bool:
    tasks = candidate.get("tasks")
    metadata = candidate.get("checkpoint_meta")
    if candidate.get("controller") != "policy" or not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(metadata, dict):
        return False
    hidden_size = exact_nonnegative_int(metadata.get("hidden_size"))
    return (
        metadata.get("controller") == "policy"
        and hidden_size is not None
        and hidden_size > 0
        and config_task(report.get("config")) == TASK_IDS.get(tasks[0])
        and positive_config_int(report.get("config"), "policy", "hidden_size") == hidden_size
    )


def valid_config(config: object, env_name: str) -> bool:
    return (
        isinstance(config, dict)
        and isinstance(config.get("base"), dict)
        and config["base"].get("env_name") == env_name
        and isinstance(config.get("env"), dict)
        and config_task(config) in TASK_IDS.values()
        and positive_config_int(config, "policy", "hidden_size") is not None
    )


def positive_config_int(config: object, section: str, key: str) -> int | None:
    if not isinstance(config, dict) or not isinstance(config.get(section), dict):
        return None
    raw = config[section].get(key)
    if isinstance(raw, str) and raw.isdecimal():
        raw = int(raw)
    parsed = exact_nonnegative_int(raw)
    return parsed if parsed is not None and parsed > 0 else None


def config_task(config: object) -> int | None:
    if not isinstance(config, dict) or not isinstance(config.get("env"), dict):
        return None
    raw = config["env"].get("task_id")
    if isinstance(raw, str) and raw.isdecimal():
        raw = int(raw)
    return exact_nonnegative_int(raw)


def valid_export_file(value: object) -> bool:
    if not isinstance(value, dict) or value.get("exists") is not True:
        return False
    size = exact_nonnegative_int(value.get("bytes"))
    lines = exact_nonnegative_int(value.get("lines"))
    return size is not None and size > 0 and lines is not None and lines > 0
