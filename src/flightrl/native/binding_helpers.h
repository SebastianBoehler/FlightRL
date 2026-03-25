#ifndef FLIGHTRL_BINDING_HELPERS_H
#define FLIGHTRL_BINDING_HELPERS_H

#include <Python.h>
#include <numpy/arrayobject.h>

#include "native_env.h"

static int flightrl_set_float(PyObject *dict, const char *key, float value) {
    PyObject *obj = PyFloat_FromDouble(value);
    if (obj == NULL) {
        return -1;
    }
    if (PyDict_SetItemString(dict, key, obj) < 0) {
        Py_DECREF(obj);
        return -1;
    }
    Py_DECREF(obj);
    return 0;
}

static double flightrl_unpack_number(PyObject *kwargs, const char *key) {
    PyObject *value = PyDict_GetItemString(kwargs, key);
    if (value == NULL) {
        PyErr_Format(PyExc_TypeError, "Missing keyword argument '%s'", key);
        return 0.0;
    }
    if (PyLong_Check(value)) {
        return (double)PyLong_AsLong(value);
    }
    if (PyFloat_Check(value)) {
        return PyFloat_AsDouble(value);
    }
    PyErr_Format(PyExc_TypeError, "Keyword '%s' must be numeric", key);
    return 0.0;
}

static DronePlanarEnv *flightrl_unpack_env(PyObject *args) {
    PyObject *handle = PyTuple_GetItem(args, 0);
    if (!PyLong_Check(handle)) {
        PyErr_SetString(PyExc_TypeError, "env handle must be an integer");
        return NULL;
    }
    return (DronePlanarEnv *)PyLong_AsVoidPtr(handle);
}

typedef struct {
    DronePlanarEnv **envs;
    int num_envs;
} VecEnv;

static VecEnv *flightrl_unpack_vecenv(PyObject *args) {
    PyObject *handle = PyTuple_GetItem(args, 0);
    if (!PyLong_Check(handle)) {
        PyErr_SetString(PyExc_TypeError, "vec env handle must be an integer");
        return NULL;
    }
    return (VecEnv *)PyLong_AsVoidPtr(handle);
}

#endif
