from __future__ import annotations

import json
import subprocess
import sys


def test_report_edge_budget_uses_the_executable_actor() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/report_edge_budget.py"],
        check=True,
        capture_output=True,
        text=True,
    )

    budget = json.loads(result.stdout)

    assert budget["parameter_count"] == 17_602
    assert budget["quantized_parameter_bytes"] == 18_688
    assert budget["macs_per_step"] == 96_048
    assert budget["measurement_boundary"] == (
        "static_graph_estimate_not_gap8_elf_or_latency"
    )
