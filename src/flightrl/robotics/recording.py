"""Bounded asynchronous MCAP recording of raw sensors, observations and actions."""

import json
from io import BytesIO
from queue import Full, Queue
from threading import Thread

import numpy as np
from mcap.reader import make_reader
from mcap.writer import CompressionType, Writer


class RunRecorder:
    def __init__(self, folder, identity, description):
        self.path = folder / "run.mcap"
        self.queue = Queue(maxsize=64)
        self.error = None
        self.closed = False
        self.count = 0
        self.queue_peak = 0
        self.identity, self.description = identity, description
        self.thread = Thread(target=self._run, name="robotics-recorder", daemon=True)
        self.thread.start()

    def submit(
        self, topic, payload, capture_ns, available_ns, sequence, encoding="json"
    ):
        if self.error:
            raise RuntimeError(f"Recording failed: {self.error}") from self.error
        if self.closed:
            raise RuntimeError("Recording is already finalized")
        try:
            self.queue.put_nowait(
                (topic, payload, capture_ns, available_ns, sequence, encoding)
            )
            self.queue_peak = max(self.queue_peak, self.queue.qsize())
        except Full as error:
            raise RuntimeError(
                "Recording queue full; stopped instead of dropping evidence"
            ) from error

    def frames(self, kind, frames, state, available_ns):
        for robot, levels in zip(("drone", "rover", "arm"), frames):
            for level, (rgb, depth) in enumerate(levels):
                self.submit(
                    f"/{robot}/{kind}/rgbd/{level}",
                    (rgb, depth),
                    state["capture_time_ns"],
                    available_ns,
                    state["sequence"],
                    "flightrl.rgbd.npz",
                )

    def _run(self):
        try:
            with self.path.open("wb") as output:
                writer = Writer(
                    output, compression=CompressionType.ZSTD, chunk_size=4 * 1024 * 1024
                )
                writer.start(library="FlightRL robotics recording v2")
                writer.add_metadata(
                    "run",
                    {
                        "identity": json.dumps(self.identity),
                        "scene": json.dumps(self.description),
                        "clock": "simulation nanoseconds; epoch=episode reset",
                    },
                )
                schema = writer.register_schema(
                    "FlightRL event v2", "jsonschema", b'{"type":"object"}'
                )
                channels = {}
                while True:
                    item = self.queue.get()
                    try:
                        if item is None:
                            break
                        topic, payload, capture, available, sequence, encoding = item
                        if topic not in channels:
                            channels[topic] = writer.register_channel(
                                topic,
                                encoding,
                                schema if encoding == "json" else 0,
                                {
                                    "acquisition_time": "JSON capture_time_ns or NPZ capture_time_ns",
                                    "depth": "float32 metric ray range, metres; NaN means missing",
                                    "frame": topic.split("/")[1] + "/camera_x_forward",
                                },
                            )
                        if encoding == "json":
                            data = json.dumps(payload, allow_nan=False).encode()
                        else:
                            buffer = BytesIO()
                            np.savez(
                                buffer,
                                rgb=payload[0],
                                depth=payload[1],
                                capture_time_ns=np.int64(capture),
                            )
                            data = buffer.getvalue()
                        writer.add_message(
                            channels[topic], available, data, available, sequence
                        )
                        self.count += 1
                    finally:
                        self.queue.task_done()
                writer.finish()
        except Exception as error:
            self.error = error

    def finish(self):
        if not self.closed:
            while self.thread.is_alive():
                try:
                    self.queue.put(None, timeout=0.1)
                    break
                except Full:
                    continue
            self.thread.join(timeout=30)
            self.closed = True
        if self.error:
            raise RuntimeError(f"Recording failed: {self.error}") from self.error
        if self.thread.is_alive():
            raise RuntimeError("Recording did not finalize within 30 seconds")


def replay_capture(path, sequence, available_ns):
    """Return one recorded capture and exact RGB pixels; never rerender old evidence."""
    import base64

    import cv2

    state, images = None, {}
    with path.open("rb") as source:
        reader = make_reader(source)
        for _, channel, message in reader.iter_messages(
            topics=["/capture"], start_time=available_ns, end_time=available_ns + 1
        ):
            if message.sequence == sequence:
                state = json.loads(message.data)
                break
    if state is None:
        raise ValueError("Requested capture is not in this episode")
    with path.open("rb") as source:
        topics = [f"/{r}/raw/rgbd/0" for r in ("drone", "rover", "arm")]
        for _, channel, message in make_reader(source).iter_messages(
            topics=topics,
            start_time=state["available_time_ns"],
            end_time=state["available_time_ns"] + 1,
        ):
            if message.sequence != sequence:
                continue
            with np.load(BytesIO(message.data)) as sample:
                ok, png = cv2.imencode(
                    ".png", cv2.cvtColor(sample["rgb"], cv2.COLOR_RGB2BGR)
                )
                if not ok:
                    raise RuntimeError("Replay image encoding failed")
                images[channel.topic.split("/")[1]] = base64.b64encode(png).decode()
    if len(images) != len(state["camera_poses"]):
        raise ValueError("Recorded capture is missing camera streams")
    return dict(type="replay", state=state, images=images)
