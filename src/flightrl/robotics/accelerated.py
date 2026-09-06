"""Same physics and camera contract, with virtual time advanced after sensor delivery."""

import asyncio
import json
import time


async def accelerated_episode(socket, session):
    request = 0
    while not session.paused:
        state = session.state()
        session.pending = (request, state, time.perf_counter())
        await socket.send(json.dumps(dict(type="state", state=state)))
        await socket.send(json.dumps(dict(type="capture", id=request, state=state)))
        while True:
            raw = await asyncio.wait_for(socket.recv(), 10)
            if isinstance(raw, bytes):
                await socket.send(json.dumps(session.receive(raw)))
                break
            message = json.loads(raw)
            if message["type"] == "camera":
                session.meta = message
            elif message["type"] == "display":
                session.display.append(message)
            elif message["type"] == "pause":
                session.paused = True
                session.stop_reason = "Paused by operator"
                return
            elif message["type"] == "link":
                session.mission.link = bool(message["enabled"])
                session.link_events.append(
                    dict(
                        time_s=session.world.ticks * session.world.dt,
                        enabled=session.mission.link,
                    )
                )
            else:
                raise ValueError("Unknown accelerated workbench message")
        # Five 20ms control updates contain 50 physical 2ms substeps.
        for _ in range(5):
            if not session.paused:
                session.step()
        request += 1
    await socket.send(
        json.dumps(
            dict(
                type="metrics",
                count=session.count,
                done=True,
                handover=session.mission.handover,
                sensor_valid=session.sensor_rig.valid
                if session.sensor_rig
                else [True, True],
                status="Verified mission complete"
                if session.mission.status()["success"]
                else session.stop_reason,
            )
        )
    )
