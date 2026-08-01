from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from flightrl.mujoco import is_mujoco_available
from flightrl.navigation.spatial_memory import MAP_CHANNELS


def test_semantic_vision_env_has_no_oracle_target_vector_when_available() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    from flightrl.mujoco.semantic_vision_env import MuJoCoSemanticVisionEnv

    env = MuJoCoSemanticVisionEnv(num_envs=2, seed=11)
    try:
        observations, _ = env.reset(seed=11)
        command = observations[:, env.layout.command_slice]
        maps = (
            observations[:, env.layout.map_slice]
            .reshape(
                2,
                *env.memory_config.shape,
            )
            .copy()
        )
        next_observations, rewards, terminals, truncations, _ = env.step(
            np.zeros((2, 4), dtype=np.float32)
        )
        env.record_target_observation(
            0,
            bearing_rad=0.0,
            distance_m=1.0,
            confidence=0.75,
        )
        env._write_observations()
        target_map = env.observations[0, env.layout.map_slice].reshape(
            env.memory_config.shape
        )
    finally:
        env.close()

    assert observations.shape == (2, env.layout.flat_dim)
    assert np.allclose(command.sum(axis=1), 1.0)
    assert np.count_nonzero(command) == 2
    assert maps[:, MAP_CHANNELS.index("visited")].sum() > 0
    assert maps[:, MAP_CHANNELS.index("target_evidence")].sum() == 0
    assert target_map[MAP_CHANNELS.index("target_evidence")].max() == np.float32(0.75)
    assert np.isfinite(next_observations).all()
    assert np.isfinite(rewards).all()
    assert terminals.shape == truncations.shape == (2,)


def test_active_target_only_changes_command_observation_when_available() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    from flightrl.mujoco.semantic_vision_env import MuJoCoSemanticVisionEnv

    env = MuJoCoSemanticVisionEnv(num_envs=1, seed=23)
    try:
        env.reset(seed=23)
        random_state = deepcopy(env.camera_rng.bit_generator.state)
        env.target_category_indices[0] = 0
        env.encoders[0].reset()
        env._write_observations()
        door = env.observations.copy()

        env.camera_rng.bit_generator.state = random_state
        env.target_category_indices[0] = 1
        env.encoders[0].reset()
        env._write_observations()
        monitor = env.observations.copy()
    finally:
        env.close()

    door[:, env.layout.command_slice] = 0.0
    monitor[:, env.layout.command_slice] = 0.0
    assert np.array_equal(door, monitor)


def test_camera_randomness_does_not_change_future_target_assignment_when_available() -> (
    None
):
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    from flightrl.mujoco.semantic_vision_env import MuJoCoSemanticVisionEnv

    rendered = MuJoCoSemanticVisionEnv(num_envs=2, seed=59)
    untouched = MuJoCoSemanticVisionEnv(num_envs=2, seed=59)
    mask = np.ones(2, dtype=bool)
    try:
        for _ in range(3):
            rendered._write_observations()
        rendered._assign_targets(mask)
        untouched._assign_targets(mask)
    finally:
        rendered.close()
        untouched.close()

    assert np.array_equal(
        rendered.target_category_indices,
        untouched.target_category_indices,
    )
    assert np.array_equal(
        rendered.sim.target_position,
        untouched.sim.target_position,
    )


def test_semantic_vision_policy_preserves_recurrent_state_when_available() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    from flightrl.mujoco.semantic_vision_env import MuJoCoSemanticVisionEnv
    from flightrl.mujoco.semantic_vision_policy import SemanticVisionPolicy

    env = MuJoCoSemanticVisionEnv(num_envs=2, seed=13)
    try:
        observations, _ = env.reset(seed=13)
        policy = SemanticVisionPolicy(env, hidden_size=32)
        state = policy.initial_state(2, "cpu")
        distribution, values, next_state = policy.forward_eval(
            torch.from_numpy(observations),
            state,
        )
    finally:
        env.close()

    assert distribution.mean.shape == (2, 4)
    assert values.shape == (2, 1)
    assert next_state[0].shape == (1, 2, 32)
    assert not torch.equal(state[0], next_state[0])


def test_semantic_policy_cannot_translate_without_target_evidence_when_available() -> (
    None
):
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    from flightrl.mujoco.semantic_vision_env import MuJoCoSemanticVisionEnv
    from flightrl.mujoco.semantic_observation import (
        GROUNDING_CONFIDENCE_INDEX,
        GROUNDING_HORIZONTAL_ERROR_INDEX,
    )
    from flightrl.mujoco.semantic_vision_policy import SemanticVisionPolicy

    env = MuJoCoSemanticVisionEnv(num_envs=1, seed=15)
    try:
        observations, _ = env.reset(seed=15)
        policy = SemanticVisionPolicy(env, hidden_size=32)
        with torch.no_grad():
            policy.decoder.decoder_mean.weight.zero_()
            policy.decoder.decoder_mean.bias.fill_(0.5)
        without_target, _ = policy(torch.from_numpy(observations.copy()))
        env.record_target_observation(
            0,
            bearing_rad=0.0,
            distance_m=1.0,
            confidence=0.8,
        )
        env._write_observations()
        grounded = env.observations.copy()
        state = grounded[:, env.layout.proprioception_slice]
        state[:, GROUNDING_CONFIDENCE_INDEX] = 0.8
        state[:, GROUNDING_HORIZONTAL_ERROR_INDEX] = -0.4
        with_target, _ = policy(torch.from_numpy(grounded))
    finally:
        env.close()

    assert torch.equal(without_target.mean[0, :3], torch.zeros(3))
    assert float(without_target.mean[0, 3].detach()) == pytest.approx(20.0 / 60.0)
    assert torch.allclose(with_target.mean[0, :3], torch.full((3,), 0.5))
    assert float(with_target.mean[0, 3].detach()) == pytest.approx(8.0 / 60.0)


def test_semantic_policy_sequence_forward_matches_puffer_contract_when_available() -> (
    None
):
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    from flightrl.mujoco.semantic_vision_env import MuJoCoSemanticVisionEnv
    from flightrl.mujoco.semantic_vision_policy import SemanticVisionPolicy

    env = MuJoCoSemanticVisionEnv(num_envs=2, seed=17)
    try:
        observations, _ = env.reset(seed=17)
        policy = SemanticVisionPolicy(env, hidden_size=32)
        sequence = torch.from_numpy(np.stack((observations, observations), axis=1))
        distribution, values = policy(sequence)
    finally:
        env.close()

    assert distribution.mean.shape == (4, 4)
    assert values.shape == (2, 2)


def test_simulated_semantic_detector_populates_deployment_map_when_available() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    from flightrl.mujoco.semantic_training import (
        SemanticTrainingEnv,
        SimulatedSemanticDetectorConfig,
    )

    env = SemanticTrainingEnv(
        room_seed=41,
        num_envs=1,
        seed=41,
        detector=SimulatedSemanticDetectorConfig(
            horizontal_fov_deg=360.0,
            vertical_fov_deg=180.0,
            dropout_probability=0.0,
        ),
    )
    try:
        observations, _ = env.reset(41)
        target_map = observations[0, env.layout.map_slice].reshape(
            env.memory_config.shape
        )
        expert = env.expert_actions()
    finally:
        env.close()

    assert env.target_observed[0]
    assert target_map[MAP_CHANNELS.index("target_evidence")].max() > 0.0
    assert np.isfinite(expert).all()
    assert expert.shape == (1, 4)


def test_semantic_expert_searches_without_translation_when_available() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    from flightrl.mujoco.semantic_training import (
        SemanticTrainingEnv,
        SimulatedSemanticDetectorConfig,
    )

    env = SemanticTrainingEnv(
        room_seed=43,
        num_envs=1,
        seed=43,
        detector=SimulatedSemanticDetectorConfig(dropout_probability=1.0),
    )
    try:
        env.reset(43)
        expert = env.expert_actions()
    finally:
        env.close()

    assert not env.target_observed[0]
    assert np.array_equal(expert[0, :3], np.zeros(3, dtype=np.float32))
    assert expert[0, 3] == pytest.approx(20.0 / 60.0)


def test_semantic_puffer_driver_accepts_action_pointer_when_available() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    from flightrl.mujoco.semantic_puffer_driver import SemanticPufferDriver

    driver = SemanticPufferDriver(
        room_seeds=(51, 52),
        agents_per_room=1,
        seed=51,
    )
    try:
        actions = np.ascontiguousarray(driver.expert_actions(), dtype=np.float32)
        driver.cpu_step(actions.ctypes.data)
    finally:
        driver.close()

    assert driver.observations.shape == (2, driver.obs_size)
    assert np.isfinite(driver.observations).all()
    assert driver.rewards.shape == driver.terminals.shape == (2,)


def test_semantic_teacher_step_skips_observation_rendering_when_available(
    monkeypatch,
) -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    from flightrl.mujoco.semantic_puffer_driver import SemanticPufferDriver

    driver = SemanticPufferDriver(
        room_seeds=(53,),
        agents_per_room=1,
        seed=53,
    )

    def fail_render() -> None:
        raise AssertionError("teacher evaluation rendered an unused observation")

    try:
        monkeypatch.setattr(driver.envs[0].backend, "_write_observations", fail_render)
        driver.teacher_step(driver.expert_actions())
    finally:
        driver.close()


def test_semantic_mujoco_resets_clear_of_generated_obstacles_when_available() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    from flightrl.mujoco.env import MuJoCoCrazyflieEnv
    from flightrl.mujoco.semantic_reset import SEMANTIC_RESET_CLEARANCE_M
    from flightrl.navigation.room_config import SemanticRoomGenerationConfig
    from flightrl.navigation.room_generation import generate_semantic_room

    scene = generate_semantic_room(
        10_731,
        SemanticRoomGenerationConfig.for_profile("diverse"),
    )
    env = MuJoCoCrazyflieEnv(num_envs=4, seed=10_731, semantic_scene=scene)

    for _ in range(32):
        env.reset()
        assert env.room.contains(
            env.position,
            margin=SEMANTIC_RESET_CLEARANCE_M,
        ).all()


def test_setpoint_yaw_command_respects_configured_rate_limit_when_available() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    from flightrl.mujoco.semantic_vision_env import MuJoCoSemanticVisionEnv
    from flightrl.mujoco.setpoint_control import firmware_setpoint_actions

    env = MuJoCoSemanticVisionEnv(num_envs=1, seed=29)
    try:
        low_level = firmware_setpoint_actions(
            env.sim,
            np.asarray(((0.0, 0.0, 0.0, 1.0),), dtype=np.float32),
            env.control,
        )
    finally:
        env.close()

    expected = np.deg2rad(env.control.max_yawrate_deg_s) / env.sim.max_rate[2]
    assert low_level[0, 3] == pytest.approx(expected)


def test_semantic_gym_adapter_wraps_with_pufferlib_when_available() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    puffer_emulation = pytest.importorskip("pufferlib.emulation")
    from flightrl.mujoco.semantic_gym_env import MuJoCoSemanticVisionGymEnv

    gym_env = MuJoCoSemanticVisionGymEnv(seed=31)
    env = puffer_emulation.GymnasiumPufferEnv(env=gym_env)
    try:
        observations, _ = env.reset(seed=31)
        next_observations, rewards, terminals, truncations, _ = env.step(
            np.zeros((1, 4), dtype=np.float32)
        )
    finally:
        env.close()

    assert observations.shape == next_observations.shape
    assert observations.shape[-1] == gym_env.backend.layout.flat_dim
    assert np.isfinite(rewards).all()
    assert np.asarray(terminals).shape in ((), (1,))
    assert np.asarray(truncations).shape in ((), (1,))
