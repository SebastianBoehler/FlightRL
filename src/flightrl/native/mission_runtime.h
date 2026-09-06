#ifndef FLIGHTRL_MISSION_RUNTIME_H
#define FLIGHTRL_MISSION_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FLIGHTRL_MISSION_ABI_VERSION 1u
#define FLIGHTRL_MISSION_NO_PHASE UINT32_MAX

enum FlightRLMissionStatus {
    FLIGHTRL_MISSION_OK = 0,
    FLIGHTRL_MISSION_INVALID_ARGUMENT = 1,
    FLIGHTRL_MISSION_INVALID_TRANSITION = 2,
    FLIGHTRL_MISSION_INCOMPATIBLE_ABI = 3,
};

enum FlightRLMissionPhase {
    FLIGHTRL_MISSION_PREFLIGHT = 0,
    FLIGHTRL_MISSION_TAKEOFF = 1,
    FLIGHTRL_MISSION_SEARCH = 2,
    FLIGHTRL_MISSION_VERIFY = 3,
    FLIGHTRL_MISSION_NAVIGATE = 4,
    FLIGHTRL_MISSION_RECOVER = 5,
    FLIGHTRL_MISSION_HOLD = 6,
    FLIGHTRL_MISSION_LAND = 7,
    FLIGHTRL_MISSION_ABORT = 8,
    FLIGHTRL_MISSION_PHASE_COUNT = 9,
};

enum FlightRLMissionEvent {
    FLIGHTRL_EVENT_PREFLIGHT_PASSED = 0,
    FLIGHTRL_EVENT_TAKEOFF_READY = 1,
    FLIGHTRL_EVENT_TARGET_ACQUIRED = 2,
    FLIGHTRL_EVENT_TARGET_CONFIRMED = 3,
    FLIGHTRL_EVENT_TARGET_REJECTED = 4,
    FLIGHTRL_EVENT_TARGET_LOST = 5,
    FLIGHTRL_EVENT_BLOCKED = 6,
    FLIGHTRL_EVENT_RECOVERED = 7,
    FLIGHTRL_EVENT_GOAL_REACHED = 8,
    FLIGHTRL_EVENT_LANDING_REQUESTED = 9,
    FLIGHTRL_EVENT_LANDED = 10,
    FLIGHTRL_EVENT_TIMEOUT = 11,
    FLIGHTRL_EVENT_ABORT_REQUESTED = 12,
    FLIGHTRL_EVENT_COUNT = 13,
};

enum FlightRLCommandSource {
    FLIGHTRL_COMMAND_PREFLIGHT = 0,
    FLIGHTRL_COMMAND_CONTROLLER = 1,
    FLIGHTRL_COMMAND_POLICY = 2,
    FLIGHTRL_COMMAND_PERCEPTION = 3,
    FLIGHTRL_COMMAND_ABORT = 4,
};

typedef struct FlightRLMissionState {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t phase;
    uint32_t resume_phase;
    uint64_t step;
} FlightRLMissionState;

typedef struct FlightRLMissionPhaseLimits {
    float max_speed_m_s;
    float max_yawrate_deg_s;
    uint8_t learned_policy_phase_eligible;
    uint8_t command_source;
    uint16_t reserved;
} FlightRLMissionPhaseLimits;

int flightrl_mission_next(FlightRLMissionState *state, uint32_t event);
int flightrl_mission_phase_limits(
    uint32_t phase,
    FlightRLMissionPhaseLimits *limits
);

#ifdef __cplusplus
}
#endif

#endif
