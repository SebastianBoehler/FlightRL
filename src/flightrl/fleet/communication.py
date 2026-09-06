"""Time-delayed peer messages; no access to simulator truth or raw peer images."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Message:
    sender: int
    timestamp: float
    position: tuple
    velocity: tuple
    target: int
    completed: tuple


class PeerLink:
    def __init__(self, count, delay_s=0.2, ttl_s=1.0, drop_probability=0.0, seed=0):
        if (
            count < 2
            or delay_s < 0
            or ttl_s <= delay_s
            or not 0 <= drop_probability <= 1
        ):
            raise ValueError("invalid peer link parameters")
        self.count, self.delay, self.ttl, self.drop = (
            count,
            delay_s,
            ttl_s,
            drop_probability,
        )
        self.rng = np.random.default_rng(seed)
        self.pending = []
        self.inbox = [dict() for _ in range(count)]
        self.delivered = 0

    def publish(self, now, estimates, velocities, targets, completed):
        for sender in range(self.count):
            msg = Message(
                sender,
                now,
                tuple(estimates[sender]),
                tuple(velocities[sender]),
                int(targets[sender]),
                tuple(completed[sender]),
            )
            for receiver in range(self.count):
                if sender != receiver and self.rng.random() >= self.drop:
                    self.pending.append((now + self.delay, receiver, msg))

    def receive(self, now):
        remaining = []
        for arrival, receiver, msg in self.pending:
            if arrival <= now:
                self.inbox[receiver][msg.sender] = msg
                self.delivered += 1
            else:
                remaining.append((arrival, receiver, msg))
        self.pending = remaining
        return [
            [m for m in box.values() if now - m.timestamp <= self.ttl]
            for box in self.inbox
        ]


def peer_features(messages, position, velocity, now):
    # Fixed two nearest peers: relative estimated pose/velocity, age, validity.
    ordered = sorted(
        messages, key=lambda m: np.linalg.norm(np.array(m.position) - position)
    )[:2]
    result = np.zeros((2, 8), np.float32)
    for i, msg in enumerate(ordered):
        result[i] = np.r_[
            np.array(msg.position) - position,
            np.array(msg.velocity) - velocity,
            now - msg.timestamp,
            1,
        ]
    return result.ravel()
