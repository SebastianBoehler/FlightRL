"""Collect measured visual-control features for both robots across seeded utility sites."""

import argparse
import asyncio
import json
from pathlib import Path
import mujoco as mj
import numpy as np
from websockets.asyncio.server import serve
from flightrl.robotics.session import RobotSession
from flightrl.robotics.sensing import decode
from flightrl.robotics.visual_control import (
    inspect_marker,
    features,
    teacher,
    supervise,
)


async def collect(socket, args, finished):
    try:
        for split, seeds, per_seed in (
            ("train", range(120, 126), 250),
            ("validation", range(140, 142), 150),
        ):
            rows = []
            for seed in seeds:
                session = RobotSession(
                    args.output / f"scene-{seed}", seed, industry=True
                )
                world = session.world
                rng = np.random.default_rng(seed * 173)
                await socket.send(
                    json.dumps(
                        dict(
                            type="scene",
                            description=session.description,
                            state=session.state(),
                            label=f"Collecting {split} industrial sensor observations",
                        )
                    )
                )
                while json.loads(await socket.recv())["type"] != "ready":
                    pass
                for i in range(per_seed + 1):
                    goals = []
                    for robot, address in ((0, 0), (1, 7)):
                        target = (
                            session.targets[0]
                            if robot == 0
                            else session.targets[2 if i % 2 else 1]
                        )
                        goals.append(target["id"])
                        near = i % 3 == 0
                        stand = 1.05 if robot == 0 else 0.55
                        distance = (
                            rng.uniform(stand - 0.12, stand + 0.45)
                            if near
                            else rng.uniform(stand + 0.3, 6.5)
                        )
                        direction = np.array(target["approach"])
                        position = np.array(target["position"]) + direction * (
                            distance + (0.035 if robot == 0 else 0.22)
                        )
                        position[1] += rng.uniform(-0.4, 0.4)
                        position[2] = (
                            position[2] + rng.uniform(-0.3, 0.3) if robot == 0 else 0.17
                        )
                        world.data.qpos[address : address + 3] = position
                        delta = np.array(target["position"]) - position
                        yaw = np.arctan2(delta[1], delta[0]) + rng.uniform(-0.22, 0.22)
                        angles = np.array(
                            [
                                rng.uniform(-0.05, 0.05) if robot == 0 else 0,
                                rng.uniform(-0.07, 0.07) if robot == 0 else 0,
                                yaw,
                            ]
                        )
                        mj.mju_euler2Quat(
                            world.data.qpos[address + 3 : address + 7], angles, "xyz"
                        )
                    world.data.qvel[:6] = rng.uniform(-0.15, 0.15, 6)
                    world.data.qvel[-2:] = rng.uniform(-3, 3, 2)
                    world.ticks = i * 50
                    mj.mj_forward(world.model, world.data)
                    session.sensor_rig.update(world, 0.1)
                    state = session.state()
                    state["goals"] = goals
                    await socket.send(json.dumps(dict(type="state", state=state)))
                    await socket.send(
                        json.dumps(dict(type="capture", id=i, state=state))
                    )
                    meta = None
                    while True:
                        raw = await asyncio.wait_for(socket.recv(), 15)
                        if isinstance(raw, bytes):
                            if meta != i:
                                raise ValueError("Mismatched collection sequence")
                            break
                        message = json.loads(raw)
                        if message["type"] == "camera":
                            meta = message["id"]
                        elif message["type"] == "pause":
                            raise RuntimeError("Collection paused by operator")
                    delivery = session.sensor_rig.deliver(decode(raw), state)
                    if delivery is None:
                        continue
                    frames, captured = delivery
                    for robot, kind in enumerate(("drone", "rover")):
                        rgb, depth = frames[robot][0]
                        valid = np.isfinite(depth).mean() > 0.5
                        measurement = (
                            inspect_marker(rgb, depth, captured["goals"][robot])
                            if valid
                            else None
                        )
                        observation = features(
                            measurement,
                            depth,
                            np.array(captured["proprio"][robot], np.float32),
                            kind,
                        )
                        command, _ = teacher(measurement, depth, kind)
                        command = supervise(command, depth, kind, valid)
                        rows.append(
                            dict(
                                features=observation,
                                action=command,
                                robot=robot,
                                seed=seed,
                                sequence=captured["sequence"],
                            )
                        )
                    if i % 25 == 0:
                        await socket.send(
                            json.dumps(
                                dict(
                                    type="metrics",
                                    count=len(rows) // 2,
                                    status=f"Actual cameras → measured features · {split} · {len(rows)} robot observations",
                                )
                            )
                        )
                print(split, seed, len(rows), flush=True)
            await asyncio.to_thread(
                np.savez_compressed,
                args.output / f"{split}.npz",
                **{k: np.array([r[k] for r in rows]) for k in rows[0]},
            )
        await socket.send(json.dumps(dict(type="saved", path=str(args.output))))
        finished.set_result(True)
    except Exception as error:
        if not finished.done():
            finished.set_exception(error)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = dict(
        train=list(range(120, 126)),
        validation=[140, 141],
        test=list(range(160, 168)),
        training_robot_observations=3000,
        validation_robot_observations=600,
        actor_inputs="Detected marker range/bearing, three depth-clearance sectors, modeled proprioception, robot role; no truth poses or asset coordinates",
        selection="Validation loss only; fixed test seeds unused until actor frozen",
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
    finished = asyncio.get_running_loop().create_future()
    async with serve(
        lambda ws: collect(ws, args, finished),
        "127.0.0.1",
        8767,
        origins=["http://127.0.0.1:4173"],
        max_size=8 * 1024 * 1024,
        compression=None,
    ):
        print("Industry collection ready", flush=True)
        await finished


if __name__ == "__main__":
    asyncio.run(main())
