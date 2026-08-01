#include "native_door_self_mask.h"

void flightrl_mask_door_airframe(
    uint8_t *frame,
    int width,
    int height
) {
    unsigned long sum = 0;
    int pixels = width * height;
    for (int index = 0; index < pixels; ++index) {
        sum += frame[index];
    }
    uint8_t fill = (uint8_t)(sum / (unsigned long)pixels);
    for (int row = 0; row < height; ++row) {
        float y = ((float)row + 0.5f) / height;
        for (int col = 0; col < width; ++col) {
            float x = ((float)col + 0.5f) / width;
            int left = (x < 0.39f && y < 0.24f)
                || (x < 0.27f && y < 0.42f);
            int right = (x > 0.67f && y < 0.24f)
                || (x > 0.75f && y < 0.40f);
            if (left || right) {
                frame[row * width + col] = fill;
            }
        }
    }
}
