#include <math.h>
#include <string.h>

#include "native_door_detector.h"

static float detector_unit(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return (float)(*state >> 8) / 16777216.0f;
}

static float detector_uniform(uint32_t *state, float lo, float hi) {
    return lo + (hi - lo) * detector_unit(state);
}

static float detector_clamp(float value, float lo, float hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

void flightrl_door_detector_reset(FlightRLDoorDetector *detector) {
    memset(detector, 0, sizeof(*detector));
    detector->recovery_yaw = 0.70f;
}

void flightrl_door_detector_teacher_action(
    FlightRLDoorDetector *detector,
    float *action
) {
    int detected = (
        detector->evidence[0] > 0.0f
        && detector->evidence[4] < 1.0f
    );
    if (!detected) {
        action[0] = 0.0f;
        action[1] = detector->target_seen
            ? detector->recovery_yaw
            : 0.85f;
        return;
    }
    float x = detector->evidence[1];
    action[1] = detector_clamp(-1.6f * x, -1.0f, 1.0f);
    if (fabsf(action[1]) > 0.15f) {
        detector->recovery_yaw = action[1] > 0.0f ? 0.55f : -0.55f;
    }
    int centered = fabsf(x) < 0.25f;
    float scale = detector->evidence[3];
    action[0] = centered && scale < 0.78f
        ? detector_clamp(1.6f * (0.78f - scale), 0.15f, 0.72f)
        : 0.0f;
}

void flightrl_door_detector_update(
    FlightRLDoorDetector *detector,
    const float *grounding,
    int control_step,
    uint32_t *rng,
    float control_dt_s,
    float maximum_evidence_age_s
) {
    if (
        !isfinite(control_dt_s)
        || !isfinite(maximum_evidence_age_s)
        || control_dt_s <= 0.0f
        || maximum_evidence_age_s <= 0.0f
    ) {
        memset(
            detector->evidence,
            0,
            sizeof(float) * (SIXDOF_DOOR_EVIDENCE_DIM - 1)
        );
        detector->evidence[4] = 1.0f;
        return;
    }
    int due = control_step >= detector->next_update_step;
    if (due) {
        int truth_visible = grounding[0] > 0.5f;
        float probability = truth_visible ? 0.85f : 0.03f;
        int detected = detector_unit(rng) < probability;
        if (detected) {
            detector->evidence[0] = truth_visible
                ? detector_uniform(rng, 0.55f, 1.0f)
                : detector_uniform(rng, 0.25f, 0.60f);
            detector->evidence[1] = truth_visible
                ? detector_clamp(
                    2.0f * grounding[1] - 1.0f
                        + detector_uniform(rng, -0.12f, 0.12f),
                    -1.0f,
                    1.0f
                )
                : detector_uniform(rng, -1.0f, 1.0f);
            detector->evidence[2] = truth_visible
                ? detector_clamp(
                    2.0f * grounding[2] - 1.0f
                        + detector_uniform(rng, -0.12f, 0.12f),
                    -1.0f,
                    1.0f
                )
                : detector_uniform(rng, -0.8f, 0.8f);
            detector->evidence[3] = truth_visible
                ? detector_clamp(
                    grounding[3] * detector_uniform(rng, 0.82f, 1.18f),
                    0.0f,
                    1.0f
                )
                : detector_uniform(rng, 0.08f, 0.40f);
            detector->target_seen = 1;
        } else {
            memset(
                detector->evidence,
                0,
                sizeof(float) * (SIXDOF_DOOR_EVIDENCE_DIM - 1)
            );
        }
        detector->last_update_step = control_step;
        detector->next_update_step = control_step + (
            6 + (int)(13.0f * detector_unit(rng))
        );
    }
    int age_steps = control_step - detector->last_update_step;
    int maximum_age_steps = (int)ceilf(
        maximum_evidence_age_s / control_dt_s
    );
    detector->evidence[4] = detector_clamp(
        (float)age_steps * control_dt_s / maximum_evidence_age_s,
        0.0f,
        1.0f
    );
    if (age_steps >= maximum_age_steps) {
        memset(
            detector->evidence,
            0,
            sizeof(float) * (SIXDOF_DOOR_EVIDENCE_DIM - 1)
        );
    }
}
