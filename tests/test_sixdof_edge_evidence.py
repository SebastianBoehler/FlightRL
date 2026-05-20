from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_sixdof_edge_evidence", ROOT / "scripts" / "build_sixdof_edge_evidence.py")
assert SPEC and SPEC.loader
EVIDENCE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVIDENCE
SPEC.loader.exec_module(EVIDENCE)


def test_select_records_includes_best_multitask_and_labels() -> None:
    args = type("Args", (), {"include_best_by_task": False, "include_best_multitask": True, "label": ["single"]})()
    selected = EVIDENCE.select_records(matrix(), args)

    assert [record["label"] for record in selected] == ["multi", "single"]


def test_matrix_args_pairs_parity_and_latency() -> None:
    record = EVIDENCE.evidence_record({"label": "multi", "checkpoint": "multi.pt", "tasks": ["a", "b"]}, args())

    assert EVIDENCE.matrix_args([record]) == [
        "--parity",
        "multi=artifacts/edge/sixdof_multi.parity.json",
        "--latency",
        "multi=artifacts/edge/sixdof_multi.latency.json",
    ]


def test_edge_evidence_dry_run_writes_manifest(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    output = tmp_path / "edge_manifest.json"
    matrix_path.write_text(json.dumps(matrix()))

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_sixdof_edge_evidence.py"),
            "--matrix",
            str(matrix_path),
            "--include-best-multitask",
            "--report",
            str(output),
            "--output-dir",
            str(tmp_path / "edge"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text())
    assert report["run"] is False
    assert report["records"][0]["label"] == "multi"
    assert "--parity" in report["matrix_args"]
    assert output.with_suffix(".md").exists()


def matrix() -> dict:
    return {
        "records": [candidate("single", ["position_yaw"]), candidate("multi", ["position_yaw", "circle"])],
        "best_by_task": {"position_yaw": candidate("single", ["position_yaw"])},
        "best_multitask": candidate("multi", ["position_yaw", "circle"]),
    }


def candidate(label: str, tasks: list[str]) -> dict:
    return {"label": label, "checkpoint": f"{label}.pt", "tasks": tasks}


def args():
    return type("Args", (), {"output_dir": "artifacts/edge", "samples": 64, "iterations": 1000, "warmup": 50})()
