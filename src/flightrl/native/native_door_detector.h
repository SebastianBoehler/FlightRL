#ifndef FLIGHTRL_NATIVE_DOOR_DETECTOR_H
#define FLIGHTRL_NATIVE_DOOR_DETECTOR_H

#include <stdint.h>

#define SIXDOF_DOOR_EVIDENCE_DIM 5

typedef struct {
    float evidence[SIXDOF_DOOR_EVIDENCE_DIM];
    int last_update_step;
    int next_update_step;
    float recovery_yaw;
    unsigned char target_seen;
} FlightRLDoorDetector;

void flightrl_door_detector_reset(FlightRLDoorDetector *detector);

void flightrl_door_detector_update(
    FlightRLDoorDetector *detector,
    const float *grounding,
    int control_step,
    uint32_t *rng,
    float control_dt_s,
    float maximum_evidence_age_s
);

void flightrl_door_detector_teacher_action(
    FlightRLDoorDetector *detector,
    float *action
);

#endif
