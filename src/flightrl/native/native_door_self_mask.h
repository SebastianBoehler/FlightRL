#ifndef FLIGHTRL_NATIVE_DOOR_SELF_MASK_H
#define FLIGHTRL_NATIVE_DOOR_SELF_MASK_H

#include <stdint.h>

void flightrl_mask_door_airframe(
    uint8_t *frame,
    int width,
    int height
);

#endif
