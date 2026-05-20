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
    assert {variant.policy_hidden_size for variant in variants} >= {64, 128, 256}
    assert {variant.replay_ratio for variant in variants} >= {1, 2}


def test_puffer_sweep_dry_run_writes_manifest(tmp_path: Path) -> None:
    output = tmp_path / "sweep.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_sixdof_puffer_sweep.py"),
            "--max-variants",
            "2",
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
    assert "--policy-hidden-size" in report["records"][0]["command"]
    assert output.with_suffix(".md").exists()
