#ifndef FLIGHTRL_NATIVE_ENV_H
#define FLIGHTRL_NATIVE_ENV_H

#include "native_rng.h"
#include "native_types.h"

void flightrl_apply_action(DronePlanarEnv *env, const float *raw_action);
void flightrl_apply_dynamics(DronePlanarEnv *env);
void flightrl_fill_observation(DronePlanarEnv *env);
void flightrl_finalize_episode(DronePlanarEnv *env, int reason);
void flightrl_record_step(DronePlanarEnv *env);
void flightrl_reset_episode_stats(DronePlanarEnv *env);
void flightrl_reset_state(DronePlanarEnv *env);
void flightrl_sample_runtime(DronePlanarEnv *env);
float flightrl_task_distance(DronePlanarEnv *env);
int flightrl_task_step(DronePlanarEnv *env);
void flightrl_update_reward(DronePlanarEnv *env, int event_code);
int flightrl_check_termination(DronePlanarEnv *env);

void c_close(DronePlanarEnv *env);
void c_reset(DronePlanarEnv *env);
void c_step(DronePlanarEnv *env);

#endif
