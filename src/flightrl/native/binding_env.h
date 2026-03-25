#ifndef FLIGHTRL_BINDING_ENV_H
#define FLIGHTRL_BINDING_ENV_H

#include "binding_helpers.h"

static int my_get(PyObject *dict, DronePlanarEnv *env);
static int my_init(DronePlanarEnv *env, PyObject *kwargs);

static PyObject *env_init(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    if (PyTuple_Size(args) != 6) {
        PyErr_SetString(PyExc_TypeError, "env_init requires six positional arguments");
        return NULL;
    }

    DronePlanarEnv *env = calloc(1, sizeof(DronePlanarEnv));
    if (env == NULL) {
        PyErr_SetString(PyExc_MemoryError, "failed to allocate environment");
        return NULL;
    }

    env->observations = PyArray_DATA((PyArrayObject *)PyTuple_GetItem(args, 0));
    env->actions = PyArray_DATA((PyArrayObject *)PyTuple_GetItem(args, 1));
    env->rewards = PyArray_DATA((PyArrayObject *)PyTuple_GetItem(args, 2));
    env->terminals = PyArray_DATA((PyArrayObject *)PyTuple_GetItem(args, 3));
    env->truncations = PyArray_DATA((PyArrayObject *)PyTuple_GetItem(args, 4));

    if (kwargs == NULL) {
        kwargs = PyDict_New();
    } else {
        Py_INCREF(kwargs);
    }

    PyObject *seed = PyTuple_GetItem(args, 5);
    PyDict_SetItemString(kwargs, "seed", seed);
    if (my_init(env, kwargs) < 0) {
        Py_DECREF(kwargs);
        free(env);
        return NULL;
    }
    Py_DECREF(kwargs);
    return PyLong_FromVoidPtr(env);
}

static PyObject *env_reset(PyObject *self, PyObject *args) {
    (void)self;
    DronePlanarEnv *env = flightrl_unpack_env(args);
    if (env == NULL) {
        return NULL;
    }
    c_reset(env);
    Py_RETURN_NONE;
}

static PyObject *env_step(PyObject *self, PyObject *args) {
    (void)self;
    DronePlanarEnv *env = flightrl_unpack_env(args);
    if (env == NULL) {
        return NULL;
    }
    c_step(env);
    Py_RETURN_NONE;
}

static PyObject *env_get(PyObject *self, PyObject *args) {
    (void)self;
    DronePlanarEnv *env = flightrl_unpack_env(args);
    if (env == NULL) {
        return NULL;
    }
    PyObject *dict = PyDict_New();
    if (dict == NULL || my_get(dict, env) < 0) {
        Py_XDECREF(dict);
        return NULL;
    }
    return dict;
}

static PyObject *env_close(PyObject *self, PyObject *args) {
    (void)self;
    DronePlanarEnv *env = flightrl_unpack_env(args);
    if (env == NULL) {
        return NULL;
    }
    c_close(env);
    free(env);
    Py_RETURN_NONE;
}

static PyObject *vectorize(PyObject *self, PyObject *args) {
    (void)self;
    int count = (int)PyTuple_Size(args);
    VecEnv *vec = calloc(1, sizeof(VecEnv));
    if (vec == NULL) {
        PyErr_SetString(PyExc_MemoryError, "failed to allocate vector env");
        return NULL;
    }
    vec->num_envs = count;
    vec->envs = calloc((size_t)count, sizeof(DronePlanarEnv *));
    if (vec->envs == NULL) {
        free(vec);
        PyErr_SetString(PyExc_MemoryError, "failed to allocate vector env handles");
        return NULL;
    }

    for (int i = 0; i < count; ++i) {
        vec->envs[i] = (DronePlanarEnv *)PyLong_AsVoidPtr(PyTuple_GetItem(args, i));
    }
    return PyLong_FromVoidPtr(vec);
}

#endif
