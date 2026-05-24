from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark_sixdof_training_throughput", ROOT / "scripts" / "benchmark_sixdof_training_throughput.py")
assert SPEC and SPEC.loader
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)


def test_default_variants_cover_training_throughput_knobs() -> None:
    variants = BENCH.default_variants()

    assert {variant.num_envs for variant in variants} >= {64, 256, 512}
    assert {variant.horizon for variant in variants} >= {16, 32, 64}
    assert {variant.hidden_size for variant in variants} >= {64, 128, 256}


def test_select_variants_preserves_order() -> None:
    selected = BENCH.select_variants(BENCH.default_variants(), ["large_512x32_h128", "smoke_64x16_h64"])

    assert [variant.name for variant in selected] == ["large_512x32_h128", "smoke_64x16_h64"]


def test_summarize_ranks_total_sps() -> None:
    variants = [BENCH.asdict(variant) for variant in BENCH.default_variants()[:2]]

    summary = BENCH.summarize(
        [
            {"variant": variants[0], "collect_sps": 100.0, "total_sps": 90.0},
            {"variant": variants[1], "collect_sps": 80.0, "total_sps": 120.0},
        ]
    )

    assert summary["best_collect_sps"]["name"] == variants[0]["name"]
    assert summary["best_total_sps"]["name"] == variants[1]["name"]


def test_training_throughput_cli_smoke(tmp_path: Path) -> None:
    output = tmp_path / "throughput.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "benchmark_sixdof_training_throughput.py"),
            "--variants",
            "smoke_64x16_h64",
            "--output",
            str(output),
            "--no-native-step",
            "--no-warmup",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())

    assert report["records"][0]["samples"] == 1024
    assert report["records"][0]["total_sps"] > 0.0
    assert output.with_suffix(".md").exists()


def test_training_throughput_cli_supports_residual_controller(tmp_path: Path) -> None:
    output = tmp_path / "residual_throughput.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "benchmark_sixdof_training_throughput.py"),
            "--variants",
            "smoke_64x16_h64",
            "--output",
            str(output),
            "--task",
            "circle",
            "--controller",
            "teacher_residual",
            "--residual-scale",
            "0.05",
            "--no-warmup",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())

    assert report["controller"] == "teacher_residual"
    assert report["residual_scale"] == 0.05
    assert report["tasks"] == ["circle"]
    assert report["records"][0]["samples"] == 1024
