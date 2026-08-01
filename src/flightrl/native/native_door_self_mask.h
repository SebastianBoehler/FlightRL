#ifndef FLIGHTRL_NATIVE_DOOR_SELF_MASK_H
#define FLIGHTRL_NATIVE_DOOR_SELF_MASK_H

#include <stdint.h>

int flightrl_door_airframe_pixel_masked(
    int row,
    int col,
    int width,
    int height
);

void flightrl_mask_door_airframe(
    uint8_t *frame,
    int width,
    int height
);

#endif
