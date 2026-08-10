from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from flightrl.exploration.mujoco_env import MuJoCoCoverageEnv
from flightrl.exploration.teacher import ScanAdvanceTeacher
from flightrl.navigation.room_generation import (
    SemanticRoomGenerationConfig,
    generate_semantic_room,
)
from flightrl.sixdof.orientation import quat_to_yaw


WIDTH, HEIGHT = 640, 480
CONTROL_HZ, VIDEO_FPS = 50, 10
FRAME_STRIDE = CONTROL_HZ // VIDEO_FPS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the simulation-only scan-advance-turn patrol from above."
    )
    parser.add_argument("--seed", type=int, default=515)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evidence/scan_advance_demo_seed515.mp4"),
    )
    args = parser.parse_args(argv)
    if args.seed < 0 or args.steps <= 0:
        parser.error("seed must be non-negative and steps must be positive")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("demo output or its metadata sidecar already exists")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("ffmpeg is required to render the demo")

    scene = generate_semantic_room(
        args.seed,
        SemanticRoomGenerationConfig.for_profile("diverse"),
    )
    env = MuJoCoCoverageEnv(scene, seed=args.seed, maximum_episode_steps=args.steps)
    external, camera, option, vehicle_geoms, vehicle_colors = _external_view(env)
    teacher = ScanAdvanceTeacher()
    observation, info = env.reset(seed=args.seed)
    teacher.reset(
        env.sim.position[0, :2],
        yaw_rad=float(quat_to_yaw(env.sim.quaternion)[0]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder = _encoder(ffmpeg, args.output)
    terminal = truncated = False
    frames = 0
    try:
        for step in range(args.steps):
            if step % FRAME_STRIDE == 0:
                _write_frame(
                    encoder,
                    external,
                    env,
                    camera,
                    option,
                    vehicle_geoms,
                    vehicle_colors,
                    phase=teacher.phase,
                    turns=teacher.completed_turns,
                )
                frames += 1
            action = teacher.action(
                env.sim.position[0, :2],
                yaw_rad=float(quat_to_yaw(env.sim.quaternion)[0]),
                horizontal_ranges_m=env.sim.ranges_m[0, :4],
            )
            observation, _reward, terminal, truncated, info = env.step(action)
            if terminal or truncated:
                break
        _write_frame(
            encoder,
            external,
            env,
            camera,
            option,
            vehicle_geoms,
            vehicle_colors,
            phase=teacher.phase,
            turns=teacher.completed_turns,
        )
        frames += 1
    finally:
        external.close()
        env.close()
        if encoder.stdin is not None:
            encoder.stdin.close()
        returncode = encoder.wait(timeout=15)
    if returncode != 0:
        raise SystemExit(f"ffmpeg exited with status {returncode}")

    report = {
        "schema": "flightrl.scan_advance_demo_video.v1",
        "seed": args.seed,
        "steps": step + 1,
        "frames": frames,
        "video_fps": VIDEO_FPS,
        "completed_turns": teacher.completed_turns,
        "collision": bool(info["collision"]),
        "boundary_violation": bool(info["boundary_violation"]),
        "coverage_score": float(info["coverage_score"]),
        "privileged_teacher": True,
        "learned_policy": False,
        "deployment_authority": False,
        "flight_authority": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not terminal else 2


def _external_view(env: MuJoCoCoverageEnv):
    mujoco = env.sim.mujoco
    ceiling = mujoco.mj_name2id(env.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "ceiling")
    env.sim.model.geom_group[ceiling] = 5
    renderer = mujoco.Renderer(env.sim.model, height=HEIGHT, width=WIDTH)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    room = env.scene.room
    camera.lookat[:] = (
        0.5 * (room.minimum[0] + room.maximum[0]),
        0.5 * (room.minimum[1] + room.maximum[1]),
        0.0,
    )
    camera.distance, camera.azimuth, camera.elevation = 6.0, 90.0, -90.0
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[5] = 0
    geoms = np.flatnonzero(env.sim.model.geom_bodyid == env.sim.body_id)
    return renderer, camera, option, geoms, env.sim.model.geom_rgba[geoms].copy()


def _encoder(ffmpeg: str, output: Path) -> subprocess.Popen:
    return subprocess.Popen(
        (
            ffmpeg, "-loglevel", "error", "-n", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}",
            "-framerate", str(VIDEO_FPS), "-i", "-", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(output),
        ),
        stdin=subprocess.PIPE,
    )


def _write_frame(encoder, renderer, env, camera, option, geoms, colors, *, phase, turns):
    env.sim.model.geom_rgba[geoms] = (1.0, 0.18, 0.02, 1.0)
    renderer.update_scene(env.sim.data[0], camera=camera, scene_option=option)
    frame = renderer.render().copy()
    env.sim.model.geom_rgba[geoms] = colors
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 210, 48), fill=(0, 0, 0))
    draw.text((20, 18), f"{phase.upper()}  turns={turns}", fill=(255, 255, 255))
    if encoder.stdin is None:
        raise RuntimeError("ffmpeg input pipe is unavailable")
    encoder.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())


if __name__ == "__main__":
    raise SystemExit(main())
