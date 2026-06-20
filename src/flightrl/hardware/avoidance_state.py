from __future__ import annotations

import math
from dataclasses import dataclass, field

from flightrl.hardware.avoidance_policy import AvoidanceCommand, RangerReading, min_horizontal_range_m, min_horizontal_ttc_s


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


@dataclass(slots=True)
class DirectionHoldState:
    hold_s: float = 0.0
    min_speed_m_s: float = 0.12
    hard_clearance_m: float = 0.11
    hard_ttc_s: float = 0.25
    _until_s: float = field(default=0.0, init=False)
    _held_vx: float = field(default=0.0, init=False)
    _held_vy: float = field(default=0.0, init=False)

    def update(
        self,
        command: AvoidanceCommand,
        *,
        now_s: float,
        reading: RangerReading,
        range_rate: RangerReading | None,
    ) -> tuple[AvoidanceCommand, bool]:
        speed = math.hypot(command.vx_m_s, command.vy_m_s)
        if self.hold_s <= 0.0 or speed < self.min_speed_m_s:
            self._until_s = 0.0
            return command, False
        if self._hard_override(reading, range_rate):
            return self._capture(command, now_s), False
        held_speed = math.hypot(self._held_vx, self._held_vy)
        if now_s < self._until_s and held_speed >= self.min_speed_m_s:
            scale = speed / held_speed
            return (
                AvoidanceCommand(
                    vx_m_s=self._held_vx * scale,
                    vy_m_s=self._held_vy * scale,
                    yawrate_deg_s=command.yawrate_deg_s,
                    zdistance_m=command.zdistance_m,
                ),
                True,
            )
        return self._capture(command, now_s), False

    def _capture(self, command: AvoidanceCommand, now_s: float) -> AvoidanceCommand:
        self._held_vx = command.vx_m_s
        self._held_vy = command.vy_m_s
        self._until_s = now_s + self.hold_s
        return command

    def _hard_override(self, reading: RangerReading, range_rate: RangerReading | None) -> bool:
        if self.hard_clearance_m > 0.0 and min_horizontal_range_m(reading) <= self.hard_clearance_m:
            return True
        return self.hard_ttc_s > 0.0 and min_horizontal_ttc_s(reading, range_rate) <= self.hard_ttc_s
