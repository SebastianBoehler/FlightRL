#ifndef FLIGHTRL_BINDING_SIXDOF_H
#define FLIGHTRL_BINDING_SIXDOF_H

#include "binding_helpers.h"
#include "native_sixdof.h"

static PyArrayObject *sixdof_array(PyObject *obj) {
    return (PyArrayObject *)PyArray_FROM_OTF(obj, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY);
}

static PyArrayObject *sixdof_read_float_array(PyObject *obj) {
    return (PyArrayObject *)PyArray_FROM_OTF(obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
}

static PyArrayObject *sixdof_int_array(PyObject *obj) {
    return (PyArrayObject *)PyArray_FROM_OTF(obj, NPY_INT32, NPY_ARRAY_INOUT_ARRAY);
}

static PyArrayObject *sixdof_uint8_array(PyObject *obj) {
    return (PyArrayObject *)PyArray_FROM_OTF(obj, NPY_UINT8, NPY_ARRAY_INOUT_ARRAY);
}

static PyObject *sixdof_step(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *pos_obj;
    PyObject *vel_obj;
    PyObject *quat_obj;
    PyObject *rates_obj;
    PyObject *ranges_obj;
    PyObject *actions_obj;
    PyObject *room_obj;
    float dt;
    if (!PyArg_ParseTuple(args, "OOOOOOOf", &pos_obj, &vel_obj, &quat_obj, &rates_obj, &ranges_obj, &actions_obj, &room_obj, &dt)) {
        return NULL;
    }

    PyArrayObject *pos = sixdof_array(pos_obj);
    PyArrayObject *vel = sixdof_array(vel_obj);
    PyArrayObject *quat = sixdof_array(quat_obj);
    PyArrayObject *rates = sixdof_array(rates_obj);
    PyArrayObject *ranges = sixdof_array(ranges_obj);
    PyArrayObject *actions = (PyArrayObject *)PyArray_FROM_OTF(actions_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *room = sixdof_read_float_array(room_obj);
    if (!pos || !vel || !quat || !rates || !ranges || !actions || !room) {
        goto fail;
    }
    int num_envs = (int)PyArray_DIM(pos, 0);
    int ok = PyArray_NDIM(pos) == 2 && PyArray_DIM(pos, 1) == 3 &&
        PyArray_NDIM(vel) == 2 && PyArray_DIM(vel, 0) == num_envs && PyArray_DIM(vel, 1) == 3 &&
        PyArray_NDIM(quat) == 2 && PyArray_DIM(quat, 0) == num_envs && PyArray_DIM(quat, 1) == 4 &&
        PyArray_NDIM(rates) == 2 && PyArray_DIM(rates, 0) == num_envs && PyArray_DIM(rates, 1) == 3 &&
        PyArray_NDIM(ranges) == 2 && PyArray_DIM(ranges, 0) == num_envs && PyArray_DIM(ranges, 1) == 6 &&
        PyArray_NDIM(actions) == 2 && PyArray_DIM(actions, 0) == num_envs && PyArray_DIM(actions, 1) == 4 &&
        PyArray_NDIM(room) == 1 && PyArray_DIM(room, 0) == 7;
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
        PyArray_DATA(room),
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
    Py_DECREF(room);
    Py_RETURN_NONE;

fail:
    Py_XDECREF(pos);
    Py_XDECREF(vel);
    Py_XDECREF(quat);
    Py_XDECREF(rates);
    Py_XDECREF(ranges);
    Py_XDECREF(actions);
    Py_XDECREF(room);
    return NULL;
}

static PyObject *sixdof_step_env(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *pos_obj, *vel_obj, *quat_obj, *rates_obj, *ranges_obj, *target_obj, *target_yaw_obj;
    PyObject *prev_obj, *step_count_obj, *actions_obj, *obs_obj, *rewards_obj, *terminals_obj, *truncations_obj, *room_obj;
    float dt;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOOOOOOOOf",
            &pos_obj,
            &vel_obj,
            &quat_obj,
            &rates_obj,
            &ranges_obj,
            &target_obj,
            &target_yaw_obj,
            &prev_obj,
            &step_count_obj,
            &actions_obj,
            &obs_obj,
            &rewards_obj,
            &terminals_obj,
            &truncations_obj,
            &room_obj,
            &dt
        )) {
        return NULL;
    }

    PyArrayObject *pos = sixdof_array(pos_obj);
    PyArrayObject *vel = sixdof_array(vel_obj);
    PyArrayObject *quat = sixdof_array(quat_obj);
    PyArrayObject *rates = sixdof_array(rates_obj);
    PyArrayObject *ranges = sixdof_array(ranges_obj);
    PyArrayObject *target = sixdof_array(target_obj);
    PyArrayObject *target_yaw = sixdof_array(target_yaw_obj);
    PyArrayObject *prev = sixdof_array(prev_obj);
    PyArrayObject *step_count = sixdof_int_array(step_count_obj);
    PyArrayObject *actions = sixdof_read_float_array(actions_obj);
    PyArrayObject *obs = sixdof_array(obs_obj);
    PyArrayObject *rewards = sixdof_array(rewards_obj);
    PyArrayObject *terminals = sixdof_uint8_array(terminals_obj);
    PyArrayObject *truncations = sixdof_uint8_array(truncations_obj);
    PyArrayObject *room = sixdof_read_float_array(room_obj);
    if (!pos || !vel || !quat || !rates || !ranges || !target || !target_yaw || !prev || !step_count || !actions || !obs || !rewards || !terminals || !truncations || !room) {
        goto fail;
    }
    int n = (int)PyArray_DIM(pos, 0);
    int ok = PyArray_NDIM(pos) == 2 && PyArray_DIM(pos, 1) == 3 &&
        PyArray_NDIM(vel) == 2 && PyArray_DIM(vel, 0) == n && PyArray_DIM(vel, 1) == 3 &&
        PyArray_NDIM(quat) == 2 && PyArray_DIM(quat, 0) == n && PyArray_DIM(quat, 1) == 4 &&
        PyArray_NDIM(rates) == 2 && PyArray_DIM(rates, 0) == n && PyArray_DIM(rates, 1) == 3 &&
        PyArray_NDIM(ranges) == 2 && PyArray_DIM(ranges, 0) == n && PyArray_DIM(ranges, 1) == 6 &&
        PyArray_NDIM(target) == 2 && PyArray_DIM(target, 0) == n && PyArray_DIM(target, 1) == 3 &&
        PyArray_NDIM(target_yaw) == 1 && PyArray_DIM(target_yaw, 0) == n &&
        PyArray_NDIM(prev) == 2 && PyArray_DIM(prev, 0) == n && PyArray_DIM(prev, 1) == 4 &&
        PyArray_NDIM(step_count) == 1 && PyArray_DIM(step_count, 0) == n &&
        PyArray_NDIM(actions) == 2 && PyArray_DIM(actions, 0) == n && PyArray_DIM(actions, 1) == 4 &&
        PyArray_NDIM(obs) == 2 && PyArray_DIM(obs, 0) == n && PyArray_DIM(obs, 1) == 28 &&
        PyArray_NDIM(rewards) == 1 && PyArray_DIM(rewards, 0) == n &&
        PyArray_NDIM(terminals) == 1 && PyArray_DIM(terminals, 0) == n &&
        PyArray_NDIM(truncations) == 1 && PyArray_DIM(truncations, 0) == n &&
        PyArray_NDIM(room) == 1 && PyArray_DIM(room, 0) == 7;
    if (!ok) {
        PyErr_SetString(PyExc_ValueError, "sixdof_step_env received incompatible array shapes");
        goto fail;
    }
    flightrl_sixdof_step_env_batch(
        PyArray_DATA(pos), PyArray_DATA(vel), PyArray_DATA(quat), PyArray_DATA(rates), PyArray_DATA(ranges),
        PyArray_DATA(target), PyArray_DATA(target_yaw), PyArray_DATA(prev), PyArray_DATA(step_count),
        PyArray_DATA(actions), PyArray_DATA(obs), PyArray_DATA(rewards), PyArray_DATA(terminals), PyArray_DATA(truncations), PyArray_DATA(room), n, dt
    );
    PyArray_ResolveWritebackIfCopy(pos);
    PyArray_ResolveWritebackIfCopy(vel);
    PyArray_ResolveWritebackIfCopy(quat);
    PyArray_ResolveWritebackIfCopy(rates);
    PyArray_ResolveWritebackIfCopy(ranges);
    PyArray_ResolveWritebackIfCopy(target);
    PyArray_ResolveWritebackIfCopy(target_yaw);
    PyArray_ResolveWritebackIfCopy(prev);
    PyArray_ResolveWritebackIfCopy(step_count);
    PyArray_ResolveWritebackIfCopy(obs);
    PyArray_ResolveWritebackIfCopy(rewards);
    PyArray_ResolveWritebackIfCopy(terminals);
    PyArray_ResolveWritebackIfCopy(truncations);
    Py_DECREF(pos); Py_DECREF(vel); Py_DECREF(quat); Py_DECREF(rates); Py_DECREF(ranges); Py_DECREF(target); Py_DECREF(target_yaw);
    Py_DECREF(prev); Py_DECREF(step_count); Py_DECREF(actions); Py_DECREF(obs); Py_DECREF(rewards); Py_DECREF(terminals); Py_DECREF(truncations);
    Py_DECREF(room);
    Py_RETURN_NONE;

fail:
    Py_XDECREF(pos); Py_XDECREF(vel); Py_XDECREF(quat); Py_XDECREF(rates); Py_XDECREF(ranges); Py_XDECREF(target); Py_XDECREF(target_yaw);
    Py_XDECREF(prev); Py_XDECREF(step_count); Py_XDECREF(actions); Py_XDECREF(obs); Py_XDECREF(rewards); Py_XDECREF(terminals); Py_XDECREF(truncations);
    Py_XDECREF(room);
    return NULL;
}

#endif
