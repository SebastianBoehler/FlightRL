#ifndef FLIGHTRL_BINDING_SIXDOF_SETPOINT_H
#define FLIGHTRL_BINDING_SIXDOF_SETPOINT_H

#include "native_sixdof_setpoint.h"

static PyObject *sixdof_setpoint_actions(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *velocity_obj, *quaternion_obj, *setpoint_obj, *physics_obj, *output_obj;
    float max_horizontal_speed, max_vertical_speed, velocity_gain, attitude_gain, vertical_gain;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOfffff",
            &velocity_obj,
            &quaternion_obj,
            &setpoint_obj,
            &physics_obj,
            &output_obj,
            &max_horizontal_speed,
            &max_vertical_speed,
            &velocity_gain,
            &attitude_gain,
            &vertical_gain
        )) {
        return NULL;
    }
    PyArrayObject *velocity = sixdof_read_float_array(velocity_obj);
    PyArrayObject *quaternion = sixdof_read_float_array(quaternion_obj);
    PyArrayObject *setpoint = sixdof_read_float_array(setpoint_obj);
    PyArrayObject *physics = sixdof_read_float_array(physics_obj);
    PyArrayObject *output = sixdof_array(output_obj);
    if (!velocity || !quaternion || !setpoint || !physics || !output) {
        goto fail;
    }
    int n = (int)PyArray_DIM(velocity, 0);
    int ok = PyArray_NDIM(velocity) == 2 && PyArray_DIM(velocity, 1) == 3 &&
        PyArray_NDIM(quaternion) == 2 && PyArray_DIM(quaternion, 0) == n && PyArray_DIM(quaternion, 1) == 4 &&
        PyArray_NDIM(setpoint) == 2 && PyArray_DIM(setpoint, 0) == n && PyArray_DIM(setpoint, 1) == 4 &&
        PyArray_NDIM(physics) == 2 && PyArray_DIM(physics, 0) == n && PyArray_DIM(physics, 1) == SIXDOF_PHYSICS_DIM &&
        PyArray_NDIM(output) == 2 && PyArray_DIM(output, 0) == n && PyArray_DIM(output, 1) == 4;
    if (!ok) {
        PyErr_SetString(PyExc_ValueError, "sixdof_setpoint_actions received incompatible array shapes");
        goto fail;
    }
    flightrl_sixdof_setpoint_actions_batch(
        PyArray_DATA(velocity),
        PyArray_DATA(quaternion),
        PyArray_DATA(setpoint),
        PyArray_DATA(physics),
        PyArray_DATA(output),
        n,
        max_horizontal_speed,
        max_vertical_speed,
        velocity_gain,
        attitude_gain,
        vertical_gain
    );
    PyArray_ResolveWritebackIfCopy(output);
    Py_DECREF(velocity); Py_DECREF(quaternion); Py_DECREF(setpoint); Py_DECREF(physics); Py_DECREF(output);
    Py_RETURN_NONE;
fail:
    Py_XDECREF(velocity); Py_XDECREF(quaternion); Py_XDECREF(setpoint); Py_XDECREF(physics); Py_XDECREF(output);
    return NULL;
}

#endif
