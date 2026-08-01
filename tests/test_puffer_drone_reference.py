from __future__ import annotations

import json
from pathlib import Path

from flightrl.sixdof.puffer_drone_reference import build_reference_report, render_markdown, write_report


def write_fake_pufferlib(root: Path) -> None:
    drone = root / "ocean" / "drone"
    config = root / "config"
    drone.mkdir(parents=True)
    config.mkdir()
    (drone / "dronelib.h").write_text(
        "\n".join(
            [
                "#define BASE_MASS 0.027f",
                "#define BASE_IXX 3.85e-6f",
                "#define BASE_IYY 3.85e-6f",
                "#define BASE_IZZ 5.9675e-6f",
                "#define BASE_ARM_LEN 0.0396f",
                "#define BASE_K_THRUST 3.16e-10f",
                "#define BASE_K_DRAG 0.005964552f",
                "#define BASE_GRAVITY 9.81f",
                "#define BASE_MAX_RPM 21702.0f",
                "#define BASE_K_MOT 0.15f",
                "#define BASE_MAX_VEL 20.0f",
                "#define BASE_MAX_OMEGA 20.0f",
                "#define DT 0.002f",
                "#define ACTION_SUBSTEPS 5",
                "// Tau_iner.x Tau_iner.y Tau_iner.z",
                "// rk4_step(",
            ]
        )
    )
    (drone / "binding.c").write_text("#define OBS_SIZE 21\n#define NUM_ATNS 4\n")
    (config / "drone.ini").write_text(
        "\n".join(
            [
                "[vec]",
                "total_agents = 2048",
                "num_buffers = 8",
                "num_threads = 1",
                "[env]",
                "num_drones = 64",
                "hover_frac = 0.8",
                "race_frac = 0.7",
                "sphere_frac = 0.0",
                "cube_frac = 0.0",
                "flag_frac = 0.0",
                "dr = 0.05",
                "use_rk2 = 0",
                "[policy]",
                "hidden_size = 64",
            ]
        )
    )


def test_reference_report_marks_official_drone_as_baseline_not_drop_in(tmp_path: Path) -> None:
    write_fake_pufferlib(tmp_path)

    report = build_reference_report(tmp_path)

    assert report["official_puffer_drone"]["observation_dim"] == 21
    assert len(report["official_puffer_drone"]["source_files"]["dronelib"]["sha256"]) == 64
    assert report["flightrl"]["observation_dim"] == 28
    assert report["compatibility"]["action_dim_match"] is True
    assert report["compatibility"]["adaptation_required_for_replacement"] is True
    assert "observation_contract_differs" in report["compatibility"]["replacement_blockers"]
    assert "mass_profile_differs" not in report["compatibility"]["replacement_blockers"]
    assert "motor_time_constant_differs" not in report["compatibility"]["replacement_blockers"]
    assert "angular_dynamics_equations_differ" in report["compatibility"]["replacement_blockers"]
    assert "integration_scheme_differs" in report["compatibility"]["replacement_blockers"]


def test_reference_report_markdown_and_json_outputs(tmp_path: Path) -> None:
    write_fake_pufferlib(tmp_path / "puffer")
    report = build_reference_report(tmp_path / "puffer")
    output = tmp_path / "report.json"

    write_report(report, output)

    assert (
        json.loads(output.read_text())["compatibility"]["safe_use"]
        == "official_speed_baseline_and_parameter_comparison_only"
    )
    assert "Puffer Drone Parameter Alignment (Non-Parity)" in output.with_suffix(".md").read_text()
    assert "Replacement blockers" in render_markdown(report)
