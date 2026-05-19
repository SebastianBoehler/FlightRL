#ifndef FLIGHTRL_BINDING_SIXDOF_H
#define FLIGHTRL_BINDING_SIXDOF_H

#include "binding_helpers.h"
#include "native_sixdof.h"

static PyArrayObject *sixdof_array(PyObject *obj) {
    return (PyArrayObject *)PyArray_FROM_OTF(obj, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY);
}

static PyObject *sixdof_step(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *pos_obj;
    PyObject *vel_obj;
    PyObject *quat_obj;
    PyObject *rates_obj;
    PyObject *ranges_obj;
    PyObject *actions_obj;
    float dt;
    if (!PyArg_ParseTuple(args, "OOOOOOf", &pos_obj, &vel_obj, &quat_obj, &rates_obj, &ranges_obj, &actions_obj, &dt)) {
        return NULL;
    }

    PyArrayObject *pos = sixdof_array(pos_obj);
    PyArrayObject *vel = sixdof_array(vel_obj);
    PyArrayObject *quat = sixdof_array(quat_obj);
    PyArrayObject *rates = sixdof_array(rates_obj);
    PyArrayObject *ranges = sixdof_array(ranges_obj);
    PyArrayObject *actions = (PyArrayObject *)PyArray_FROM_OTF(actions_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (!pos || !vel || !quat || !rates || !ranges || !actions) {
        goto fail;
    }
    int num_envs = (int)PyArray_DIM(pos, 0);
    int ok = PyArray_NDIM(pos) == 2 && PyArray_DIM(pos, 1) == 3 &&
        PyArray_NDIM(vel) == 2 && PyArray_DIM(vel, 0) == num_envs && PyArray_DIM(vel, 1) == 3 &&
        PyArray_NDIM(quat) == 2 && PyArray_DIM(quat, 0) == num_envs && PyArray_DIM(quat, 1) == 4 &&
        PyArray_NDIM(rates) == 2 && PyArray_DIM(rates, 0) == num_envs && PyArray_DIM(rates, 1) == 3 &&
        PyArray_NDIM(ranges) == 2 && PyArray_DIM(ranges, 0) == num_envs && PyArray_DIM(ranges, 1) == 6 &&
        PyArray_NDIM(actions) == 2 && PyArray_DIM(actions, 0) == num_envs && PyArray_DIM(actions, 1) == 4;
    if (!ok) {
        PyErr_SetString(PyExc_ValueError, "sixdof_step expects shapes position(N,3), velocity(N,3), quaternion(N,4), body_rates(N,3), ranges(N,6), actions(N,4)");
        goto fail;
    }

    flightrl_sixdof_step_batch(
        PyArray_DATA(pos),
        PyArray_DATA(vel),
        PyArray_DATA(quat),
        PyArray_DATA(rates),
        PyArray_DATA(ranges),
        PyArray_DATA(actions),
        num_envs,
        dt
    );
    PyArray_ResolveWritebackIfCopy(pos);
    PyArray_ResolveWritebackIfCopy(vel);
    PyArray_ResolveWritebackIfCopy(quat);
    PyArray_ResolveWritebackIfCopy(rates);
    PyArray_ResolveWritebackIfCopy(ranges);
    Py_DECREF(pos);
    Py_DECREF(vel);
    Py_DECREF(quat);
    Py_DECREF(rates);
    Py_DECREF(ranges);
    Py_DECREF(actions);
    Py_RETURN_NONE;

fail:
    Py_XDECREF(pos);
    Py_XDECREF(vel);
    Py_XDECREF(quat);
    Py_XDECREF(rates);
    Py_XDECREF(ranges);
    Py_XDECREF(actions);
    return NULL;
}

#endif
