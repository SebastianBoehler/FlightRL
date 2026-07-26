from __future__ import annotations

from dataclasses import dataclass

from .physics import SixDofPhysicsProfile, resolve_physics_profile
from .puffer_evaluation import PufferEvalConfig, evaluate_puffer_mujoco, evaluate_puffer_python


@dataclass(frozen=True, slots=True)
class PhysicsSweepGrid:
    linear_drag: tuple[float, ...] = (0.04, 0.06, 0.08, 0.10)
    rate_tau_s: tuple[float, ...] = (0.035, 0.045, 0.060)
    motor_tau_s: tuple[float, ...] = (0.0, 0.035, 0.060)
    thrust_scale: tuple[float, ...] = (0.75,)


def candidate_profiles(base: str | SixDofPhysicsProfile | None, grid: PhysicsSweepGrid) -> list[SixDofPhysicsProfile]:
    base_profile = resolve_physics_profile(base)
    profiles = []
    for drag in grid.linear_drag:
        for rate_tau in grid.rate_tau_s:
            for motor_tau in grid.motor_tau_s:
                for thrust_scale in grid.thrust_scale:
                    profiles.append(
                        SixDofPhysicsProfile(
                            mass_kg=base_profile.mass_kg,
                            gravity_m_s2=base_profile.gravity_m_s2,
                            linear_drag=drag,
                            rate_tau_s=rate_tau,
                            thrust_scale=thrust_scale,
                            max_rate_rad_s=base_profile.max_rate_rad_s,
                            motor_tau_s=motor_tau,
                        )
                    )
    return profiles


def calibrate_puffer_physics(policy, config: PufferEvalConfig, profiles: list[SixDofPhysicsProfile]) -> dict:
    target = evaluate_puffer_mujoco(policy, config)
    if target.get("status") != "ok":
        return {"target": target, "records": [], "best": None, "passed": False}
    records = []
    for index, profile in enumerate(profiles):
        python = evaluate_puffer_python(policy, replace_config(config, backend="python", physics_profile=profile, domain_randomization=None))
        record = {
            "index": index,
            "physics_profile": profile.as_report(),
            "python": python,
            "score": profile_score(target["metrics"], python["metrics"]),
        }
        records.append(record)
    records.sort(key=lambda item: item["score"])
    return {"target": target, "records": records, "best": records[0] if records else None, "passed": bool(records)}


def profile_score(target: dict, candidate: dict) -> float:
    return float(
        5.0 * abs(target["open_space_horizontal_speed_p95_m_s"] - candidate["open_space_horizontal_speed_p95_m_s"])
        + 2.0 * abs(target["horizontal_speed_p95_m_s"] - candidate["horizontal_speed_p95_m_s"])
        + 2.0 * abs(target["mean_position_error_m"] - candidate["mean_position_error_m"])
        + abs(target["clearance_p01_m"] - candidate["clearance_p01_m"])
        + 0.02 * abs(target["tilt_p95_deg"] - candidate["tilt_p95_deg"])
    )


def replace_config(config: PufferEvalConfig, **updates) -> PufferEvalConfig:
    values = {
        "task": config.task,
        "backend": config.backend,
        "steps": config.steps,
        "num_envs": config.num_envs,
        "seed": config.seed,
        "reset_profile": config.reset_profile,
        "sensor_profile": config.sensor_profile,
        "physics_profile": config.physics_profile,
        "domain_randomization": config.domain_randomization,
        "disturbance_profile": config.disturbance_profile,
        "min_clearance_m": config.min_clearance_m,
        "min_completed_fraction": config.min_completed_fraction,
        "max_position_error_m": config.max_position_error_m,
        "max_horizontal_speed_p95_m_s": config.max_horizontal_speed_p95_m_s,
        "max_open_space_horizontal_speed_p95_m_s": config.max_open_space_horizontal_speed_p95_m_s,
        "max_tilt_p95_deg": config.max_tilt_p95_deg,
    }
    values.update(updates)
    return PufferEvalConfig(**values)


def render_calibration_markdown(report: dict) -> str:
    lines = ["# Puffer Physics Calibration", ""]
    target = report["target"]
    if target.get("status") != "ok":
        return "\n".join([*lines, f"Target status: `{target.get('status')}`"])
    metric = target["metrics"]
    lines.extend(
        [
            "## MuJoCo Target",
            "",
            f"- Open-space speed p95: `{metric['open_space_horizontal_speed_p95_m_s']:.3f}` m/s",
            f"- Position error: `{metric['mean_position_error_m']:.3f}` m",
            f"- Tilt p95: `{metric['tilt_p95_deg']:.1f}` deg",
            "",
            "## Ranked Python Profiles",
            "",
            "| rank | score | linear drag | rate tau | motor tau | thrust scale | open-space speed p95 | position error | tilt p95 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rank, record in enumerate(report["records"][:10], start=1):
        profile = record["physics_profile"]
        metrics = record["python"]["metrics"]
        lines.append(
            f"| {rank} | {record['score']:.4f} | {profile['linear_drag']:.3f} | {profile['rate_tau_s']:.3f} | "
            f"{profile['motor_tau_s']:.3f} | {profile['thrust_scale']:.3f} | {metrics['open_space_horizontal_speed_p95_m_s']:.3f} | "
            f"{metrics['mean_position_error_m']:.3f} | {metrics['tilt_p95_deg']:.1f} |"
        )
    return "\n".join(lines)
