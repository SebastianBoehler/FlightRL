static float clampf(float value, float low, float high) {
    return value < low ? low : (value > high ? high : value);
}

#include "../src/flightrl/native/native_door_teacher.c"
