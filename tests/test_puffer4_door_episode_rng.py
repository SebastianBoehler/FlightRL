from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
RNG_SOURCE = ROOT / "src/flightrl/native/native_door_episode_rng.c"


class EpisodeRng(ctypes.Structure):
    _fields_ = (
        ("base_seed", ctypes.c_uint32),
        ("appearance_seed", ctypes.c_uint32),
        ("env_index", ctypes.c_uint32),
        ("next_episode_index", ctypes.c_uint64),
    )


class GroupLog(ctypes.Structure):
    _fields_ = (
        ("layout_episode_fraction", ctypes.c_float * 3),
        ("layout_success_fraction", ctypes.c_float * 3),
        ("door_face_episode_fraction", ctypes.c_float * 3),
        ("door_face_success_fraction", ctypes.c_float * 3),
        ("low_light_episode_fraction", ctypes.c_float),
        ("low_light_success_fraction", ctypes.c_float),
        ("obstacle_episode_fraction", ctypes.c_float),
        ("obstacle_success_fraction", ctypes.c_float),
    )


@pytest.fixture(scope="module")
def episode_rng_library(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the episode RNG contract test")
    if not RNG_SOURCE.is_file():
        pytest.fail("native episode-indexed door RNG helper is not implemented")
    library_path = tmp_path_factory.mktemp("door-episode-rng") / "episode_rng.so"
    subprocess.run(
        (
            compiler,
            "-shared",
            "-fPIC",
            str(RNG_SOURCE),
            "-o",
            str(library_path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    library = ctypes.CDLL(str(library_path))
    library.flightrl_door_episode_rng_init.argtypes = (
        ctypes.POINTER(EpisodeRng),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    )
    library.flightrl_door_episode_rng_next.argtypes = (
        ctypes.POINTER(EpisodeRng),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    )
    library.flightrl_door_episode_rng_next.restype = ctypes.c_uint64
    library.flightrl_door_scene_group_id.argtypes = (ctypes.c_uint8,) * 5
    library.flightrl_door_scene_group_id.restype = ctypes.c_uint8
    return library


def next_seeds(library, state: EpisodeRng) -> tuple[int, int, int]:
    physical = ctypes.c_uint32()
    appearance = ctypes.c_uint32()
    episode = library.flightrl_door_episode_rng_next(
        ctypes.byref(state),
        ctypes.byref(physical),
        ctypes.byref(appearance),
    )
    return episode, physical.value, appearance.value


def initialized(library, base: int, appearance: int, env: int) -> EpisodeRng:
    state = EpisodeRng()
    library.flightrl_door_episode_rng_init(
        ctypes.byref(state),
        base,
        appearance,
        env,
    )
    return state


def test_episode_rng_matches_frozen_golden_vectors(episode_rng_library) -> None:
    state = initialized(episode_rng_library, 11, 2_003, 0)

    assert next_seeds(episode_rng_library, state) == (
        0,
        269_082_774,
        2_795_547_550,
    )
    assert next_seeds(episode_rng_library, state) == (
        1,
        875_367_926,
        3_047_567_932,
    )


def test_episode_rng_uses_env_and_high_episode_words(episode_rng_library) -> None:
    env_127 = initialized(episode_rng_library, 11, 2_003, 127)
    high_episode = initialized(episode_rng_library, 11, 2_003, 0)
    high_episode.next_episode_index = 1 << 32

    assert next_seeds(episode_rng_library, env_127) == (
        0,
        1_533_061_996,
        3_352_814_589,
    )
    assert next_seeds(episode_rng_library, high_episode) == (
        1 << 32,
        1_170_715_199,
        482_279_273,
    )


def test_appearance_seed_does_not_change_physical_stream(
    episode_rng_library,
) -> None:
    original = initialized(episode_rng_library, 11, 2_003, 0)
    changed = initialized(episode_rng_library, 11, 2_004, 0)

    assert next_seeds(episode_rng_library, original) == (
        0,
        269_082_774,
        2_795_547_550,
    )
    assert next_seeds(episode_rng_library, changed) == (
        0,
        269_082_774,
        1_820_640_715,
    )


def test_next_episode_is_independent_of_prior_rng_draw_count(
    episode_rng_library,
) -> None:
    short = initialized(episode_rng_library, 23, 2_003, 5)
    long = initialized(episode_rng_library, 23, 2_003, 5)
    _, short_physical, short_appearance = next_seeds(episode_rng_library, short)
    _, long_physical, long_appearance = next_seeds(episode_rng_library, long)

    for _ in range(3):
        short_physical = (1_664_525 * short_physical + 1_013_904_223) & 0xFFFFFFFF
    for _ in range(4_003):
        long_physical = (1_664_525 * long_physical + 1_013_904_223) & 0xFFFFFFFF
    for _ in range(2):
        short_appearance = (
            1_664_525 * short_appearance + 1_013_904_223
        ) & 0xFFFFFFFF
    for _ in range(8_002):
        long_appearance = (
            1_664_525 * long_appearance + 1_013_904_223
        ) & 0xFFFFFFFF

    assert (short_physical, short_appearance) != (
        long_physical,
        long_appearance,
    )
    assert next_seeds(episode_rng_library, short) == next_seeds(
        episode_rng_library,
        long,
    )


def test_scene_group_id_uses_frozen_compact_schema(episode_rng_library) -> None:
    group_id = episode_rng_library.flightrl_door_scene_group_id(
        3,
        2,
        1,
        0,
        1,
    )

    assert group_id == 0b0101_1011


def test_group_log_adds_only_positive_marginal_counters(
    episode_rng_library,
) -> None:
    try:
        add_group = episode_rng_library.flightrl_door_group_log_add
    except AttributeError:
        pytest.fail("native marginal door group logger is not implemented")
    add_group.argtypes = (
        ctypes.POINTER(GroupLog),
        ctypes.c_uint8,
        ctypes.c_uint8,
    )
    group_log = GroupLog()

    add_group(ctypes.byref(group_log), 0b0101_1011, 1)
    add_group(ctypes.byref(group_log), 0b0010_0000, 0)

    assert list(group_log.layout_episode_fraction) == pytest.approx((0, 0, 1))
    assert list(group_log.layout_success_fraction) == pytest.approx((0, 0, 1))
    assert list(group_log.door_face_episode_fraction) == pytest.approx((0, 1, 0))
    assert list(group_log.door_face_success_fraction) == pytest.approx((0, 1, 0))
    assert group_log.low_light_episode_fraction == pytest.approx(1)
    assert group_log.low_light_success_fraction == pytest.approx(1)
    assert group_log.obstacle_episode_fraction == pytest.approx(1)
    assert group_log.obstacle_success_fraction == pytest.approx(0)
