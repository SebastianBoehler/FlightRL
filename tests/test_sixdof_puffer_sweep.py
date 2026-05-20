from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_sixdof_puffer_sweep", ROOT / "scripts" / "run_sixdof_puffer_sweep.py")
assert SPEC and SPEC.loader
SWEEP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SWEEP
SPEC.loader.exec_module(SWEEP)


def test_parse_sps_samples_handles_puffer_suffixes() -> None:
    text = "\x1b[0;0HSPS             444.6K\nSPS               1.3M\nSPS              98"
    assert SWEEP.parse_sps_samples(text) == [444_600.0, 1_300_000.0, 98.0]


def test_parse_train_sps_samples_skips_eval_only_dashboard() -> None:
    text = (
        "\x1b[0;0H│  SPS             444.6K      Misc       0ms   0%  │\n"
        "│  Steps           655.4K    Train      598ms  68%  │\n"
        "\x1b[0;0H│  SPS               1.3M      Misc       0ms   0%  │\n"
        "│  Steps             2.4M    Train        0ms   0%  │\n"
    )
    assert SWEEP.parse_train_sps_samples(text) == [444_600.0]


def test_default_sweep_covers_size_and_policy_knobs() -> None:
    variants = SWEEP.default_variants()
    assert {variant.total_agents for variant in variants} >= {1024, 4096, 8192}
    assert {variant.num_threads for variant in variants} >= {1, 4, 8, 12}
    assert {variant.horizon for variant in variants} >= {16, 32, 64}
    assert {variant.policy_hidden_size for variant in variants} >= {64, 128, 256}
    assert {variant.replay_ratio for variant in variants} >= {1, 2}


def test_select_variants_keeps_requested_order() -> None:
    variants = SWEEP.select_variants(SWEEP.default_variants(), ["large_h32_t12_rr1_h128", "fast_h32_t4_rr1_h128"])

    assert [variant.name for variant in variants] == ["large_h32_t12_rr1_h128", "fast_h32_t4_rr1_h128"]


def test_summarize_finds_fastest_completed_train_sps() -> None:
    variants = [SWEEP.asdict(variant) for variant in SWEEP.default_variants()[:2]]
    summary = SWEEP.summarize(
        [
            {"variant": variants[0], "returncode": 0, "max_train_sps": 100.0, "elapsed_s": 1.0},
            {"variant": variants[1], "returncode": 0, "max_train_sps": 250.0, "elapsed_s": 2.0},
        ]
    )

    assert summary["completed"] == 2
    assert summary["best_train_sps"]["name"] == variants[1]["name"]
    assert summary["best_train_sps"]["max_train_sps"] == 250.0


def test_render_markdown_marks_failed_run_instead_of_pending() -> None:
    variant = SWEEP.default_variants()[0]
    report = {"records": [{"variant": SWEEP.asdict(variant), "returncode": 2, "max_train_sps": None}]}

    markdown = SWEEP.render_markdown(report)

    assert "failed rc=2" in markdown
    assert "pending" not in markdown


def test_tail_lines_keeps_only_recent_lines() -> None:
    assert SWEEP.tail_lines("a\nb\nc", limit=2) == ["b", "c"]


def test_puffer_sweep_dry_run_writes_manifest(tmp_path: Path) -> None:
    output = tmp_path / "sweep.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_sixdof_puffer_sweep.py"),
            "--max-variants",
            "2",
            "--variants",
            "fast_h32_t4_rr1_h128",
            "large_h32_t12_rr1_h128",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["run"] is False
    assert len(report["records"]) == 2
    assert report["summary"]["total"] == 2
    assert [record["variant"]["name"] for record in report["records"]] == ["fast_h32_t4_rr1_h128", "large_h32_t12_rr1_h128"]
    assert "--policy-hidden-size" in report["records"][0]["command"]
    assert output.with_suffix(".md").exists()
