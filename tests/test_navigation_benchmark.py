from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from flightrl.navigation.benchmark import build_navigation_benchmark_report, evaluate_scenario_record
from flightrl.navigation.scenarios import DEFAULT_NAVIGATION_SCENARIOS, scenario_by_name


ROOT = Path(__file__).resolve().parents[1]


def test_default_navigation_scenarios_stay_single_drone_without_perception() -> None:
    names = {scenario.name for scenario in DEFAULT_NAVIGATION_SCENARIOS}

    assert {"target_approach", "obstacle_room", "vertical_clearance", "hold_or_land"} <= names
    for scenario in DEFAULT_NAVIGATION_SCENARIOS:
        assert scenario.drones_per_env == 1
        assert scenario.observation_source == "range_telemetry"
        assert "camera" not in scenario.tags
        assert "depth" not in scenario.required_metrics


def test_navigation_scenario_scores_and_reports_threshold_failures() -> None:
    scenario = scenario_by_name("obstacle_room")
    passing = evaluate_scenario_record(
        scenario,
        label="candidate",
        metrics={
            "mean_completed_fraction": 0.95,
            "mean_survival_fraction": 0.98,
            "mean_position_error_m": 0.2,
            "clearance_p01_m": 0.32,
            "action_saturation_fraction": 0.04,
        },
    )
    failing = evaluate_scenario_record(
        scenario,
        label="candidate",
        metrics={
            "mean_completed_fraction": 0.95,
            "mean_survival_fraction": 0.98,
            "mean_position_error_m": 0.2,
            "clearance_p01_m": 0.04,
            "action_saturation_fraction": 0.04,
        },
    )

    assert passing["passed"] is True
    assert passing["score"] > failing["score"]
    assert failing["passed"] is False
    assert failing["failures"] == ["clearance_p01_m_lt_0.12"]


def test_navigation_benchmark_report_ranks_candidates_by_scenario() -> None:
    report = build_navigation_benchmark_report(
        [
            record("alpha", "target_approach", completed=0.80, clearance=0.30, position_error=0.5),
            record("bravo", "target_approach", completed=0.98, clearance=0.24, position_error=0.2),
            record("alpha", "vertical_clearance", completed=0.90, clearance=0.09, position_error=0.4),
        ]
    )

    assert report["summary"]["scenarios"] == 2
    assert report["summary"]["passed_records"] == 2
    assert report["best_by_scenario"]["target_approach"]["label"] == "bravo"
    assert report["records"][0]["drones_per_env"] == 1
    assert report["safety"] == "Simulation benchmark only; not approved for live hardware."


def test_navigation_benchmark_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    input_path = tmp_path / "records.json"
    output_path = tmp_path / "navigation_benchmark.json"
    input_path.write_text(json.dumps({"records": [record("candidate", "hold_or_land")]}))

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_navigation_benchmark_report.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    saved = json.loads(output_path.read_text())
    assert "navigation_benchmark=" in result.stdout
    assert saved["records"][0]["scenario"] == "hold_or_land"
    assert output_path.with_suffix(".md").exists()


def record(
    label: str,
    scenario: str,
    *,
    completed: float = 0.95,
    survival: float = 0.97,
    clearance: float = 0.25,
    position_error: float = 0.15,
    saturation: float = 0.04,
) -> dict:
    return {
        "label": label,
        "scenario": scenario,
        "checkpoint": f"artifacts/checkpoints/{label}.pt",
        "metrics": {
            "mean_completed_fraction": completed,
            "mean_survival_fraction": survival,
            "mean_position_error_m": position_error,
            "clearance_p01_m": clearance,
            "action_saturation_fraction": saturation,
        },
    }
