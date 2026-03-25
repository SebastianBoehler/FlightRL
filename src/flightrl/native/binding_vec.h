#ifndef FLIGHTRL_BINDING_VEC_H
#define FLIGHTRL_BINDING_VEC_H

#include "binding_helpers.h"

static int my_log(PyObject *dict, Log *log);

static PyObject *vec_reset(PyObject *self, PyObject *args) {
    (void)self;
    VecEnv *vec = flightrl_unpack_vecenv(args);
    if (vec == NULL) {
        return NULL;
    }
    unsigned long seed = PyTuple_Size(args) > 1 ? PyLong_AsUnsignedLong(PyTuple_GetItem(args, 1)) : 0UL;
    for (int i = 0; i < vec->num_envs; ++i) {
        vec->envs[i]->rng_state = (uint64_t)(seed + (unsigned long)i) + 0x9e3779b97f4a7c15ULL;
        c_reset(vec->envs[i]);
    }
    Py_RETURN_NONE;
}

static PyObject *vec_step(PyObject *self, PyObject *args) {
    (void)self;
    VecEnv *vec = flightrl_unpack_vecenv(args);
    if (vec == NULL) {
        return NULL;
    }
    for (int i = 0; i < vec->num_envs; ++i) {
        c_step(vec->envs[i]);
    }
    Py_RETURN_NONE;
}

static PyObject *vec_log(PyObject *self, PyObject *args) {
    (void)self;
    VecEnv *vec = flightrl_unpack_vecenv(args);
    if (vec == NULL) {
        return NULL;
    }

    Log aggregate = {0};
    for (int i = 0; i < vec->num_envs; ++i) {
        Log *log = &vec->envs[i]->log;
        aggregate.episode_return += log->episode_return;
        aggregate.episode_length += log->episode_length;
        aggregate.success_rate += log->success_rate;
        aggregate.crash_rate += log->crash_rate;
        aggregate.timeout_rate += log->timeout_rate;
        aggregate.mean_distance += log->mean_distance;
        aggregate.mean_action_magnitude += log->mean_action_magnitude;
        aggregate.n += log->n;
        memset(log, 0, sizeof(Log));
    }

    PyObject *dict = PyDict_New();
    if (dict == NULL || aggregate.n == 0.0f) {
        return dict;
    }

    float n = aggregate.n;
    aggregate.episode_return /= n;
    aggregate.episode_length /= n;
    aggregate.success_rate /= n;
    aggregate.crash_rate /= n;
    aggregate.timeout_rate /= n;
    aggregate.mean_distance /= n;
    aggregate.mean_action_magnitude /= n;

    if (my_log(dict, &aggregate) < 0 || flightrl_set_float(dict, "n", n) < 0) {
        Py_DECREF(dict);
        return NULL;
    }
    return dict;
}

static PyObject *vec_close(PyObject *self, PyObject *args) {
    (void)self;
    VecEnv *vec = flightrl_unpack_vecenv(args);
    if (vec == NULL) {
        return NULL;
    }
    for (int i = 0; i < vec->num_envs; ++i) {
        c_close(vec->envs[i]);
        free(vec->envs[i]);
    }
    free(vec->envs);
    free(vec);
    Py_RETURN_NONE;
}

#endif
