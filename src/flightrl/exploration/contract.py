from __future__ import annotations

from typing import Any

from flightrl.puffer4_edge_schema import (
    EDGE_FRAME_PIXELS,
    EDGE_HEIGHT,
    EDGE_TELEMETRY_DIM,
    EDGE_WIDTH,
    TELEMETRY_SPECS,
)


COVERAGE_CONTRACT_ID = "aideck-coverage-policy-v1"
COVERAGE_OBSERVATION_DIM = EDGE_FRAME_PIXELS + EDGE_TELEMETRY_DIM
COVERAGE_MAXIMUM_YAW_RATE_DEG_S = 8.0


def coverage_contract_payload() -> dict[str, Any]:
    telemetry_start = EDGE_FRAME_PIXELS
    return {
        "contract_id": COVERAGE_CONTRACT_ID,
        "schema_version": 1,
        "status": "simulation_research_contract_only",
        "observation": {
            "flat_values": COVERAGE_OBSERVATION_DIM,
            "flat_dtype": "float32",
            "segments": {
                "current_gray4": [0, telemetry_start],
                "telemetry": [telemetry_start, COVERAGE_OBSERVATION_DIM],
            },
            "frame": {
                "width": EDGE_WIDTH,
                "height": EDGE_HEIGHT,
                "wire_encoding": "packed_gray4_row_major",
                "model_formula": "float32(unpacked_nibble) / 15.0",
            },
            "telemetry": {
                "values": EDGE_TELEMETRY_DIM,
                "order": [spec[0] for spec in TELEMETRY_SPECS],
                "source": "hardware_shaped_non_range_state_and_applied_action",
            },
            "prohibited_actor_inputs": [
                "target_token",
                "target_pose",
                "object_detection",
                "range_rays",
                "occupancy_grid",
                "scene_geometry",
                "privileged_pose",
            ],
            "temporal_context": "recurrent_state_only",
        },
        "action": {
            "wire_order": ["vx", "vy", "vz", "yaw_rate"],
            "controlled_axes": ["vx", "yaw_rate"],
            "structurally_zero_axes": ["vy", "vz"],
            "maximum_forward_speed_m_s": 0.25,
            "maximum_yaw_rate_deg_s": COVERAGE_MAXIMUM_YAW_RATE_DEG_S,
            "altitude_owner": "stm32_not_actor",
        },
        "objective": {
            "training_signal": "privileged_newly_visited_or_visible_free_cells",
            "privileged_grid_exposed_to_actor": False,
            "collision_and_clearance_from_simulator_truth": True,
        },
        "required_causal_gates": [
            "frame_shuffle_action_loss_increase_at_least_5_percent",
            "frozen_and_reversed_frames_reduce_closed_loop_coverage",
            "held_out_obstacle_collision_gate",
        ],
        "authority": "simulation_only",
        "shadow_authority": False,
        "deployment_authority": False,
        "flight_authority": False,
    }
