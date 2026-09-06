"""Local WebGPU sensor / Jolt simulation bridge. Run beside the existing Vite viewer."""

import argparse
import asyncio
import json
import time
from pathlib import Path
from uuid import uuid4
from websockets.asyncio.server import serve
from flightrl.realism.session import Session


async def connection(socket, actor, output):
    session = None
    tasks = []
    try:
        payload = json.loads(await asyncio.wait_for(socket.recv(), 30))
        if payload.get("type") != "scene":
            raise ValueError("Shared scene required first")
        session = Session(payload["scene"], actor, output / str(uuid4())[:8])
        await socket.send(
            json.dumps(
                {
                    "type": "ready",
                    "identity": session.identity,
                    "state": session.state(True),
                }
            )
        )

        async def physics():
            deadline = time.perf_counter()
            while True:
                deadline += 0.02
                await asyncio.sleep(max(0, deadline - time.perf_counter()))
                if time.perf_counter() - deadline > 0.5:
                    raise RuntimeError(
                        "Physics clock fell more than 500 ms behind; paused instead of changing timestep"
                    )
                if session.mode != "paused":
                    session.step()
                    await socket.send(
                        json.dumps({"type": "state", "state": session.state()})
                    )

        async def sensors():
            request = 0
            deadline = time.perf_counter()
            captured = False
            while True:
                await asyncio.sleep(max(0, deadline - time.perf_counter()))
                deadline += 0.1
                now = time.perf_counter()
                if session.pending is not None:
                    if now - session.pending[2] > 5:
                        raise RuntimeError("No completed camera batch for five seconds")
                    continue
                if session.mode == "paused" and captured:
                    continue
                state = session.state(True)
                session.pending = (request, state, now)
                await socket.send(
                    json.dumps({"type": "capture", "id": request, "state": state})
                )
                request += 1
                captured = True

        async def receive():
            async for message in socket:
                if isinstance(message, bytes):
                    result = session.receive(message)
                    await socket.send(json.dumps(result))
                    continue
                command = json.loads(message)
                if command["type"] == "camera":
                    session.camera_meta = command
                elif command["type"] == "mode":
                    session.set_mode(command["mode"])
                    await socket.send(
                        json.dumps({"type": "state", "state": session.state()})
                    )
                    if session.mode == "paused":
                        await asyncio.to_thread(session.save)
                        await socket.send(
                            json.dumps({"type": "saved", "path": str(session.folder)})
                        )
                elif command["type"] == "rain":
                    session.particles.rain = bool(command["enabled"])
                elif command["type"] == "drop":
                    session.world.drop_props()
                elif command["type"] == "display":
                    session.display.append(command)
                else:
                    raise ValueError("Unknown simulation command")

        tasks = [asyncio.create_task(f()) for f in (physics, sensors, receive)]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except Exception as error:
        print(f"Realism session failed: {error}", flush=True)
        try:
            await socket.send(json.dumps({"type": "error", "message": str(error)}))
        except Exception:
            pass
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if session:
            session.save()
            session.world.world.close()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.actor.is_file():
        raise FileNotFoundError(args.actor)
    args.output.mkdir(parents=True, exist_ok=False)
    async with serve(
        lambda ws: connection(ws, args.actor, args.output),
        "127.0.0.1",
        8766,
        origins=["http://127.0.0.1:4173"],
        max_size=32 * 1024 * 1024,
        compression=None,
    ):
        print(
            "Native realism bridge ready on 127.0.0.1:8766; open /realism.html on port 4173",
            flush=True,
        )
        await asyncio.get_running_loop().create_future()


if __name__ == "__main__":
    asyncio.run(main())
