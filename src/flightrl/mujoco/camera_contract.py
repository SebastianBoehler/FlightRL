from __future__ import annotations

from math import atan, degrees, radians, tan


AIDECK_SOURCE_WIDTH = 162
AIDECK_SOURCE_HEIGHT = 122
# Bitcraze publishes 87 degrees horizontally for the full-width HM01B0 image.
# The 162x122 QQVGA stream uses the 324x244 window, so MuJoCo's vertical FOV
# must be derived from the window aspect ratio rather than set to 87 degrees.
AIDECK_HORIZONTAL_FOV_DEG = 87.0
AIDECK_MUJOCO_VERTICAL_FOV_DEG = degrees(
    2.0
    * atan(
        tan(radians(AIDECK_HORIZONTAL_FOV_DEG) / 2.0)
        * AIDECK_SOURCE_HEIGHT
        / AIDECK_SOURCE_WIDTH
    )
)
