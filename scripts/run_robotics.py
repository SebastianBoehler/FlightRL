"""Serve the local mixed-robot inspection workbench."""

import argparse
import asyncio
import json
import time
from pathlib import Path
from uuid import uuid4
from urllib.parse import parse_qs, urlsplit
from websockets.asyncio.server import serve
from flightrl.robotics.session import RobotSession
from flightrl.robotics.policy import Policy
from flightrl.robotics.pair_policy import PairPolicy
from flightrl.robotics.accelerated import accelerated_episode
from flightrl.robotics.recording import replay_capture


async def episode(socket, args, seed):
    session = RobotSession(
        args.output / str(uuid4())[:8], seed, args.policy, args.industry, args.arm
    )
    tasks = []
    try:
        await socket.send(
            json.dumps(
                dict(
                    type="scene",
                    description=session.description,
                    state=session.state(),
                    label="MuJoCo · recorded sensor observations · simulation only",
                )
            )
        )
        message = json.loads(await asyncio.wait_for(socket.recv(), 60))
        while message["type"] == "display":
            message = json.loads(await asyncio.wait_for(socket.recv(), 60))
        if message["type"] != "ready":
            raise ValueError("Renderer must acknowledge scene")

        session.started = time.perf_counter()
        if args.fast:
            await accelerated_episode(socket, session)
            return session.folder, session.stop_reason

        async def save():
            await asyncio.to_thread(session.save)
            await socket.send(
                json.dumps(
                    dict(
                        type="saved",
                        path=str(session.folder),
                        captures=session.capture_index,
                    )
                )
            )

        async def physics():
            deadline = time.perf_counter()
            while True:
                deadline += 0.02
                await asyncio.sleep(max(0, deadline - time.perf_counter()))
                if not session.paused and time.perf_counter() - deadline > 0.5:
                    raise RuntimeError("Physics clock over 500ms late")
                if not session.paused:
                    session.step()
                    if session.paused:
                        await socket.send(
                            json.dumps(
                                dict(
                                    type="metrics",
                                    count=session.count,
                                    status=session.stop_reason,
                                    done=True,
                                    handover=session.mission.handover,
                                    sensor_valid=session.sensor_rig.valid
                                    if session.sensor_rig
                                    else [True, True],
                                )
                            )
                        )
                        if session.pending is None:
                            await save()
                        if args.seeds:
                            return
                    await socket.send(
                        json.dumps(dict(type="state", state=session.state()))
                    )

        async def sensors():
            request = 0
            deadline = time.perf_counter()
            while True:
                await asyncio.sleep(max(0, deadline - time.perf_counter()))
                deadline += 0.1
                if session.pending:
                    if time.perf_counter() - session.pending[2] > 5:
                        raise RuntimeError("Camera delivery timed out")
                    continue
                if session.paused:
                    continue
                state = session.state()
                session.pending = (request, state, time.perf_counter())
                await socket.send(
                    json.dumps(dict(type="capture", id=request, state=state))
                )
                request += 1

        async def receive():
            async for raw in socket:
                if isinstance(raw, bytes):
                    await socket.send(json.dumps(session.receive(raw)))
                    if session.paused:
                        await save()
                        if args.seeds:
                            return
                    continue
                message = json.loads(raw)
                if message["type"] == "camera":
                    session.meta = message
                elif message["type"] == "display":
                    session.display.append(message)
                elif message["type"] == "link":
                    if session.paused:
                        raise ValueError("Cannot change a finalized episode")
                    session.mission.link = bool(message["enabled"])
                    session.link_events.append(
                        dict(
                            time_s=session.world.ticks * session.world.dt,
                            enabled=session.mission.link,
                        )
                    )
                elif message["type"] == "pause":
                    session.paused = True
                    session.stop_reason = "Paused by operator"
                    if session.pending is not None:
                        continue
                    await save()
                    if args.seeds:
                        return
                elif message["type"] == "arm":
                    if session.paused or session.world.arm is None:
                        raise ValueError(
                            "Arm controls require a live articulated model"
                        )
                    session.world.arm.command(message["control"])
                elif message["type"] == "replay":
                    if not session.paused:
                        raise ValueError(
                            "Pause and finalize the recording before replay"
                        )
                    sequence = message["sequence"]
                    if type(sequence) is not int:
                        raise ValueError("Replay sequence must be an integer")
                    capture = next(
                        (c for c in session.capture_index if c["sequence"] == sequence),
                        None,
                    )
                    if capture is None:
                        raise ValueError("Unknown capture in this episode")
                    result = await asyncio.to_thread(
                        replay_capture,
                        session.recorder.path,
                        sequence,
                        capture["available_time_ns"],
                    )
                    await socket.send(json.dumps(result))
                else:
                    raise ValueError("Unknown workbench command")

        tasks = [asyncio.create_task(f()) for f in (physics, sensors, receive)]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except Exception as error:
        session.stop_reason = str(error)
        print(f"Robot session failed: {error}", flush=True)
        try:
            await socket.send(json.dumps(dict(type="error", message=str(error))))
        except Exception:
            pass
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.to_thread(session.save)

    return session.folder, session.stop_reason


async def connection(socket, args, finished):
    reports = []
    seed = args.seed
    selection = parse_qs(urlsplit(socket.request.path).query).get("site")
    if selection and args.industry and not args.seeds:
        sites = {"power": 0, "production": 1}
        if len(selection) != 1 or selection[0] not in sites:
            await socket.send(
                json.dumps(dict(type="error", message="Unknown industrial site"))
            )
            return
        seed = sites[selection[0]]
    for seed in args.seeds or [seed]:
        folder, reason = await episode(socket, args, seed)
        reports.append(json.loads((folder / "report.json").read_text()))
        if reason == "Paused by operator":
            break
    if args.seeds:
        summary = dict(
            requested_seeds=args.seeds,
            reports=reports,
            successes=sum(r["success"] for r in reports),
        )
        (args.output / "suite.json").write_text(json.dumps(summary, indent=2))
        if not finished.done():
            finished.set_result(True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actor", type=Path)
    parser.add_argument("--industry", action="store_true")
    parser.add_argument(
        "--arm", action="store_true", help="Attach the pinned xArm7 and wrist camera"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Advance virtual time after each camera batch for evaluation",
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    args.policy = (
        (PairPolicy(args.actor) if args.industry else Policy(args.actor))
        if args.actor
        else None
    )
    if args.fast and not args.seeds:
        parser.error("--fast requires an explicit --seeds evaluation suite")
    args.output.mkdir(parents=True, exist_ok=False)
    finished = asyncio.get_running_loop().create_future()
    async with serve(
        lambda ws: connection(ws, args, finished),
        "127.0.0.1",
        8767,
        origins=["http://127.0.0.1:4173"],
        max_size=8 * 1024 * 1024,
        compression=None,
    ):
        print("Robot bridge ready on 8767; open /robotics.html", flush=True)
        await finished


if __name__ == "__main__":
    asyncio.run(main())
