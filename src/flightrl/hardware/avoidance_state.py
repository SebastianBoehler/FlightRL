from __future__ import annotations

import math
from dataclasses import dataclass, field

from flightrl.hardware.avoidance_policy import AvoidanceCommand


@dataclass(slots=True)
class EscapeHoldState:
    hold_steps: int = 0
    min_speed_m_s: float = 0.05
    _remaining: int = field(default=0, init=False)
    _held_vx: float = field(default=0.0, init=False)
    _held_vy: float = field(default=0.0, init=False)

    def update(self, command: AvoidanceCommand, *, emergency: bool) -> tuple[AvoidanceCommand, bool]:
        speed = math.hypot(command.vx_m_s, command.vy_m_s)
        if not emergency or speed < self.min_speed_m_s or self.hold_steps <= 0:
            self._remaining = 0
            return command, False

        if self._remaining > 0:
            held_speed = math.hypot(self._held_vx, self._held_vy)
            if held_speed >= self.min_speed_m_s:
                scale = speed / held_speed
                self._remaining -= 1
                return (
                    AvoidanceCommand(
                        vx_m_s=self._held_vx * scale,
                        vy_m_s=self._held_vy * scale,
                        yawrate_deg_s=command.yawrate_deg_s,
                        zdistance_m=command.zdistance_m,
                    ),
                    True,
                )

        self._held_vx = command.vx_m_s
        self._held_vy = command.vy_m_s
        self._remaining = self.hold_steps
        return command, False
