"""Scripted setpoint experiment using native closed-loop dynamics, not autonomy."""

import numpy as np


class CornerDustDemo:
    label = "scripted_corner_dust"
    recovered = False
    finished = False

    def __init__(self, start):
        self.tick = -1
        self.inspected = set()
        self.panels = {}
        self.events = []
        self.mode = "approach dust corner"

    def observe(self, rgb, depth, estimate, quaternion, connected):
        self.tick += 1

    def command(self, position, quaternion):
        t = self.tick * 0.1
        if t < 10:
            goal = np.array([-3.15, -2.25, 0.65])
            phase = "approach dust corner"
        elif t < 20:
            goal = np.array([-3.15, -2.25, 0.65])
            phase = "stir dust bed"
        elif t < 24:
            goal = np.array([-3.15, -2.25, 1.0])
            phase = "climb through plume"
        else:
            goal = np.array([-1.8, -1.3, 1.5])
            phase = "retreat and observe settling"
        if phase != self.mode or self.tick == 0:
            self.events.append({"tick": self.tick, "type": phase})
        self.mode = phase
        yaw = 2 * np.arctan2(quaternion[3], quaternion[0])
        delta = goal - position
        c, s = np.cos(yaw), np.sin(yaw)
        body = np.array(
            [c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1], delta[2]]
        )
        yaw_error = np.arctan2(np.sin(-2.55 - yaw), np.cos(-2.55 - yaw))
        command = np.r_[np.clip(body * 0.8, -0.3, 0.3), np.clip(yaw_error, -0.6, 0.6)]
        return command.astype(np.float32), np.r_[body, yaw_error].astype(np.float32)
