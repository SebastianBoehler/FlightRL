from __future__ import annotations

from typing import Any


RANGE_EXPLORATION_CONTRACT_ID = "range-frontier-exploration-v2"
RANGE_MAP_SHAPE = (4, 32, 32)
RANGE_MAP_VALUES = 4 * 32 * 32
RANGE_SCALAR_VALUES = 10
RANGE_EXPLORATION_OBSERVATION_DIM = RANGE_MAP_VALUES + RANGE_SCALAR_VALUES
RANGE_ACTION_DIM = 2


def range_exploration_contract_payload() -> dict[str, Any]:
    return {
        "contract_id": RANGE_EXPLORATION_CONTRACT_ID,
        "schema_version": 2,
        "status": "simulation_research_contract_only",
        "observation": {
            "flat_values": RANGE_EXPLORATION_OBSERVATION_DIM,
            "flat_dtype": "float32",
            "segments": {
                "exploration_map": [0, 4096],
                "horizontal_ranges": [4096, 4100],
                "range_validity": [4100, 4104],
                "previous_applied_action": [4104, 4106],
            },
            "map": {
                "shape": list(RANGE_MAP_SHAPE),
                "channels": [
                    "visited",
                    "observed_free",
                    "occupied",
                    "frontier",
                ],
                "body_aligned": True,
                "span_m": 6.4,
                "output_cell_size_m": 0.2,
            },
            "horizontal_range_order": [
                "range.front",
                "range.back",
                "range.left",
                "range.right",
            ],
            "prohibited_actor_inputs": [
                "selected_frontier",
                "target_bearing",
                "target_pose",
                "scene_geometry",
                "privileged_pose",
                "simulator_truth",
            ],
            "temporal_context": "explicit_occupancy_map_and_previous_action",
        },
        "action": {
            "normalized_order": ["forward", "yaw"],
            "normalized_bounds": [[0.0, 1.0], [-1.0, 1.0]],
            "simulation_maximum_forward_speed_m_s": 0.5,
            "simulation_maximum_yaw_rate_deg_s": 90.0,
            "structurally_zero_axes": ["vy", "vz"],
            "altitude_owner": "crazyflie_firmware_and_runtime",
            "policy_owns_exploration_direction": True,
            "safety_shell_has_veto_only": True,
        },
        "objective": {
            "visited_weight": 0.35,
            "observed_free_weight": 0.65,
            "collision_penalty": -2.0,
            "selected_frontier_reward": False,
            "truth_exposed_to_actor": False,
        },
        "authority": {
            "training": False,
            "shadow": False,
            "deployment": False,
            "flight": False,
        },
    }
