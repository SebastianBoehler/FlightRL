#ifndef FLIGHTRL_BINDING_SIXDOF_VISION_H
#define FLIGHTRL_BINDING_SIXDOF_VISION_H

#include "native_sixdof_vision.h"

static PyArrayObject *sixdof_read_uint8_array(PyObject *obj) {
    return (PyArrayObject *)PyArray_FROM_OTF(obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
}

static PyObject *sixdof_render_gray4(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *position_obj, *quaternion_obj, *room_obj, *target_mean_obj, *scene_seed_obj, *frames_obj;
    if (!PyArg_ParseTuple(args, "OOOOOO", &position_obj, &quaternion_obj, &room_obj, &target_mean_obj, &scene_seed_obj, &frames_obj)) {
        return NULL;
    }
    PyArrayObject *position = sixdof_read_float_array(position_obj);
    PyArrayObject *quaternion = sixdof_read_float_array(quaternion_obj);
    PyArrayObject *room = sixdof_read_float_array(room_obj);
    PyArrayObject *target_mean = sixdof_read_float_array(target_mean_obj);
    PyArrayObject *scene_seed = (PyArrayObject *)PyArray_FROM_OTF(scene_seed_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *frames = sixdof_uint8_array(frames_obj);
    if (!position || !quaternion || !room || !target_mean || !scene_seed || !frames) {
        goto fail;
    }
    int n = (int)PyArray_DIM(position, 0);
    int ok = PyArray_NDIM(position) == 2 && PyArray_DIM(position, 1) == 3 &&
        PyArray_NDIM(quaternion) == 2 && PyArray_DIM(quaternion, 0) == n && PyArray_DIM(quaternion, 1) == 4 &&
        PyArray_NDIM(room) == 1 && PyArray_DIM(room, 0) == 7 &&
        PyArray_NDIM(target_mean) == 1 && PyArray_DIM(target_mean, 0) == n &&
        PyArray_NDIM(scene_seed) == 1 && PyArray_DIM(scene_seed, 0) == n &&
        PyArray_NDIM(frames) == 3 && PyArray_DIM(frames, 0) == n &&
        PyArray_DIM(frames, 1) == SIXDOF_VISION_HEIGHT && PyArray_DIM(frames, 2) == SIXDOF_VISION_WIDTH;
    if (!ok) {
        PyErr_SetString(PyExc_ValueError, "sixdof_render_gray4 received incompatible array shapes");
        goto fail;
    }
    flightrl_sixdof_render_gray4_batch(
        PyArray_DATA(position), PyArray_DATA(quaternion), PyArray_DATA(room), PyArray_DATA(target_mean),
        PyArray_DATA(scene_seed), PyArray_DATA(frames), n
    );
    PyArray_ResolveWritebackIfCopy(frames);
    Py_DECREF(position); Py_DECREF(quaternion); Py_DECREF(room); Py_DECREF(target_mean); Py_DECREF(scene_seed); Py_DECREF(frames);
    Py_RETURN_NONE;
fail:
    Py_XDECREF(position); Py_XDECREF(quaternion); Py_XDECREF(room); Py_XDECREF(target_mean); Py_XDECREF(scene_seed); Py_XDECREF(frames);
    return NULL;
}

static PyObject *sixdof_visual_observation(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *position_obj, *quaternion_obj, *target_obj, *target_yaw_obj, *room_obj, *target_mean_obj;
    PyObject *scene_seed_obj, *previous_obj, *reset_obj, *observation_obj;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOO", &position_obj, &quaternion_obj, &target_obj, &target_yaw_obj, &room_obj,
            &target_mean_obj, &scene_seed_obj, &previous_obj, &reset_obj, &observation_obj)) {
        return NULL;
    }
    PyArrayObject *position = sixdof_read_float_array(position_obj);
    PyArrayObject *quaternion = sixdof_read_float_array(quaternion_obj);
    PyArrayObject *target = sixdof_read_float_array(target_obj);
    PyArrayObject *target_yaw = sixdof_read_float_array(target_yaw_obj);
    PyArrayObject *room = sixdof_read_float_array(room_obj);
    PyArrayObject *target_mean = sixdof_read_float_array(target_mean_obj);
    PyArrayObject *scene_seed = (PyArrayObject *)PyArray_FROM_OTF(scene_seed_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *previous = sixdof_uint8_array(previous_obj);
    PyArrayObject *reset = sixdof_read_uint8_array(reset_obj);
    PyArrayObject *observation = sixdof_array(observation_obj);
    if (!position || !quaternion || !target || !target_yaw || !room || !target_mean || !scene_seed || !previous || !reset || !observation) {
        goto fail;
    }
    int n = (int)PyArray_DIM(position, 0);
    int ok = PyArray_NDIM(position) == 2 && PyArray_DIM(position, 1) == 3 &&
        PyArray_NDIM(quaternion) == 2 && PyArray_DIM(quaternion, 0) == n && PyArray_DIM(quaternion, 1) == 4 &&
        PyArray_NDIM(target) == 2 && PyArray_DIM(target, 0) == n && PyArray_DIM(target, 1) == 3 &&
        PyArray_NDIM(target_yaw) == 1 && PyArray_DIM(target_yaw, 0) == n &&
        PyArray_NDIM(room) == 1 && PyArray_DIM(room, 0) == 7 &&
        PyArray_NDIM(target_mean) == 1 && PyArray_DIM(target_mean, 0) == n &&
        PyArray_NDIM(scene_seed) == 1 && PyArray_DIM(scene_seed, 0) == n &&
        PyArray_NDIM(previous) == 3 && PyArray_DIM(previous, 0) == n &&
        PyArray_DIM(previous, 1) == SIXDOF_VISION_HEIGHT && PyArray_DIM(previous, 2) == SIXDOF_VISION_WIDTH &&
        PyArray_NDIM(reset) == 1 && PyArray_DIM(reset, 0) == n &&
        PyArray_NDIM(observation) == 2 && PyArray_DIM(observation, 0) == n && PyArray_DIM(observation, 1) == SIXDOF_VISION_OBS_DIM;
    if (!ok) {
        PyErr_SetString(PyExc_ValueError, "sixdof_visual_observation received incompatible array shapes");
        goto fail;
    }
    flightrl_sixdof_visual_observation_batch(
        PyArray_DATA(position), PyArray_DATA(quaternion), PyArray_DATA(target), PyArray_DATA(target_yaw), PyArray_DATA(room),
        PyArray_DATA(target_mean), PyArray_DATA(scene_seed), PyArray_DATA(previous), PyArray_DATA(reset), PyArray_DATA(observation), n
    );
    PyArray_ResolveWritebackIfCopy(previous);
    PyArray_ResolveWritebackIfCopy(observation);
    Py_DECREF(position); Py_DECREF(quaternion); Py_DECREF(target); Py_DECREF(target_yaw); Py_DECREF(room);
    Py_DECREF(target_mean); Py_DECREF(scene_seed); Py_DECREF(previous); Py_DECREF(reset); Py_DECREF(observation);
    Py_RETURN_NONE;
fail:
    Py_XDECREF(position); Py_XDECREF(quaternion); Py_XDECREF(target); Py_XDECREF(target_yaw); Py_XDECREF(room);
    Py_XDECREF(target_mean); Py_XDECREF(scene_seed); Py_XDECREF(previous); Py_XDECREF(reset); Py_XDECREF(observation);
    return NULL;
}

#endif
