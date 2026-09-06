import numpy as np
import pytest
from flightrl.inspection.environments import environment_scene
from flightrl.inspection.rollout import run_mission


@pytest.mark.parametrize("family", ["utility_plant", "data_center", "forest"])
def test_seeded_geometry_and_bound_identity(family):
    a = environment_scene(family, 1)
    b = environment_scene(family, 1)
    c = environment_scene(family, 2)
    assert a.manifest["sha256"] == b.manifest["sha256"]
    assert a.manifest["sha256"] != c.manifest["sha256"]
    assert not np.array_equal(
        a.scenario.arrays["terrain_obstacles"], c.scenario.arrays["terrain_obstacles"]
    )
    assert a.scenario.manifest["sha256"] != c.scenario.manifest["sha256"]


@pytest.mark.parametrize("size", [(128, 96), (512, 384), (768, 576)])
def test_high_resolution_sensor_is_exact_policy_source(size):
    _, _, frames, _, samples = run_mission(
        environment_scene("data_center", 1),
        industrial=True,
        ticks=2,
        sensor_size=size,
        collect=True,
    )
    w, h = size
    factor = w // 64
    assert frames.shape == (2, h, w, 3)
    for frame, sample in zip(frames, samples):
        np.testing.assert_array_equal(
            sample[0],
            frame.reshape(48, factor, 64, factor, 3).mean((1, 3)).astype(np.uint8),
        )
