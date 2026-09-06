"""Collect actual WebGPU images at seeded physical poses; labels use only pixels."""

import argparse
import asyncio
import json
from pathlib import Path
import mujoco as mj
import numpy as np
from websockets.asyncio.server import serve
from flightrl.robotics.session import RobotSession
from flightrl.robotics.sensing import decode, target_from_pixels, servo


async def collect(socket, args, finished):
    try:
        for split, seeds, per_seed in (
            ("train", [11, 12, 13, 14], 500),
            ("validation", [15, 16], 200),
        ):
            records = []
            for seed in seeds:
                session = RobotSession(args.output / f"scene-{seed}", seed)
                world = session.world
                rng = np.random.default_rng(seed * 1009)
                await socket.send(
                    json.dumps(
                        dict(
                            type="scene",
                            description=session.description,
                            state=session.state(),
                            label=f"Collecting {split} camera observations",
                        )
                    )
                )
                while json.loads(await socket.recv())["type"] != "ready":
                    pass
                for i in range(per_seed):
                    target = np.array(session.targets[0]["position"])
                    # Oversample the stopping region; no target coordinate enters labels or actor inputs.
                    near = i % 3 == 0
                    distance = rng.uniform(0.85, 1.4) if near else rng.uniform(1.1, 5.3)
                    world.data.qpos[:3] = target + [
                        -distance,
                        rng.uniform(-0.35, 0.35),
                        rng.uniform(-0.25, 0.25),
                    ]
                    angles = np.array(
                        [
                            rng.uniform(-0.07, 0.07),
                            rng.uniform(-0.1, 0.1),
                            rng.uniform(-0.22, 0.22),
                        ]
                    )
                    mj.mju_euler2Quat(world.data.qpos[3:7], angles, "xyz")
                    world.data.qvel[:6] = rng.uniform(-0.2, 0.2, 6)
                    world.ticks = i * 50
                    mj.mj_forward(world.model, world.data)
                    state = session.state()
                    await socket.send(json.dumps(dict(type="state", state=state)))
                    await socket.send(
                        json.dumps(dict(type="capture", id=i, state=state))
                    )
                    meta = None
                    while True:
                        raw = await asyncio.wait_for(socket.recv(), 10)
                        if isinstance(raw, bytes):
                            if meta != i:
                                raise ValueError("Collection camera sequence mismatch")
                            break
                        message = json.loads(raw)
                        if message["type"] == "camera":
                            meta = message["id"]
                        elif message["type"] == "pause":
                            raise RuntimeError("Collection paused by user")
                    frames = decode(raw)
                    relative = target_from_pixels(*frames[0][0], 17)
                    action, found = servo(relative, "drone")
                    rgb, depth = frames[0][1]
                    records.append(
                        dict(
                            rgb=rgb,
                            depth=depth.astype(np.float16),
                            proprio=np.array(state["proprio"][0], np.float32),
                            action=action,
                            found=found,
                            seed=seed,
                            sequence=i,
                        )
                    )
                    await socket.send(
                        json.dumps(
                            dict(
                                type="metrics",
                                count=len(records),
                                status=f"Actual WebGPU training capture · {split} · {len(records)} samples",
                            )
                        )
                    )
            await asyncio.to_thread(
                np.savez_compressed,
                args.output / f"{split}.npz",
                **{k: np.array([r[k] for r in records]) for k in records[0]},
            )
        await socket.send(json.dumps(dict(type="saved", path=str(args.output))))
        finished.set_result(True)
    except Exception as error:
        print(f"Collection failed: {error}", flush=True)
        if not finished.done():
            finished.set_exception(error)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = dict(
        train_seeds=[11, 12, 13, 14],
        validation_seeds=[15, 16],
        test_seeds=[101, 102, 103],
        train_samples=2000,
        validation_samples=400,
        labels="AprilTag RGB detection and rendered metric depth only",
        scope="Static pose imitation data in this industrial corridor; target X varies by seed, not arbitrary unseen buildings",
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
    finished = asyncio.get_running_loop().create_future()
    async with serve(
        lambda socket: collect(socket, args, finished),
        "127.0.0.1",
        8767,
        origins=["http://127.0.0.1:4173"],
        max_size=8 * 1024 * 1024,
        compression=None,
    ):
        print("Collection ready on /robotics.html", flush=True)
        await finished


if __name__ == "__main__":
    asyncio.run(main())
