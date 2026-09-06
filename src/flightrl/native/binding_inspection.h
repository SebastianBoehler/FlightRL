#ifndef FLIGHTRL_BINDING_INSPECTION_H
#define FLIGHTRL_BINDING_INSPECTION_H
#include <limits.h>
#include <math.h>
#include "inspection_scene.h"

static int inspection_array(PyObject *obj, int dtype, int ndim, int writable) {
    if (!PyArray_Check(obj)) return 0;
    PyArrayObject *a=(PyArrayObject *)obj;
    if (PyArray_TYPE(a)!=dtype || PyArray_NDIM(a)!=ndim ||
        !PyArray_IS_C_CONTIGUOUS(a) || !PyArray_ISALIGNED(a) || !PyArray_ISNOTSWAPPED(a) ||
        (writable && !PyArray_ISWRITEABLE(a))) return 0;
    for (int k=0;k<ndim;++k) if (PyArray_DIM(a,k)>INT_MAX) return 0;
    if (dtype==NPY_FLOAT32 && !writable) {
        const float *v=PyArray_DATA(a);
        for (npy_intp k=0;k<PyArray_SIZE(a);++k) if (!isfinite(v[k])) return 0;
    }
    return 1;
}
#define IDIM(a,k) PyArray_DIM((PyArrayObject *)(a),k)
#define IDATA(a) PyArray_DATA((PyArrayObject *)(a))
static PyObject *inspection_render(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *p,*q,*r,*b,*s,*f,*c,*depth=NULL,*appearance=NULL,*lights=NULL,*windows=NULL; int materials=0;
    if (!PyArg_ParseTuple(args,"OOOOOOO|OiOOO",&p,&q,&r,&b,&s,&f,&c,&depth,&materials,&appearance,&lights,&windows)) return NULL;
    if (!inspection_array(p,NPY_FLOAT32,2,0) || !inspection_array(q,NPY_FLOAT32,2,0) ||
        !inspection_array(r,NPY_FLOAT32,1,0) || !inspection_array(b,NPY_FLOAT32,2,0) ||
        !inspection_array(s,NPY_FLOAT32,2,0) || !inspection_array(f,NPY_UINT8,4,1) ||
        !inspection_array(c,NPY_INT32,3,1)) goto invalid;
    int n=(int)IDIM(p,0),m=(int)IDIM(s,0);
    if (n<1 || n>64 || m>1024 || IDIM(b,0)>1024 || IDIM(p,1)!=3 || IDIM(q,0)!=n || IDIM(q,1)!=4 ||
        IDIM(r,0)!=7 || IDIM(b,1)!=6 || IDIM(s,1)!=14 || IDIM(f,0)!=n ||
        (IDIM(f,1)!=48 && IDIM(f,1)!=96 && IDIM(f,1)!=192 && IDIM(f,1)!=384 && IDIM(f,1)!=576) || IDIM(f,2)*3!=IDIM(f,1)*4 || IDIM(f,3)!=3 || IDIM(c,0)!=n ||
        IDIM(c,1)!=m || IDIM(c,2)!=2) goto invalid;
    const float *panels=IDATA(s);
    for (int j=0;j<m;++j) for (int k=11;k<14;++k)
        if (panels[14*j+k]<0 || panels[14*j+k]>255) goto invalid;
    if (depth && (!inspection_array(depth,NPY_FLOAT32,3,1) || IDIM(depth,0)!=n ||
                  IDIM(depth,1)!=IDIM(f,1) || IDIM(depth,2)!=IDIM(f,2))) goto invalid;
    if ((materials || appearance || lights || windows) && (!appearance || !lights || !windows ||
        !inspection_array(appearance,NPY_FLOAT32,1,0) || IDIM(appearance,0)!=21 ||
        !inspection_array(lights,NPY_FLOAT32,2,0) || IDIM(lights,1)!=7 || IDIM(lights,0)>64 ||
        !inspection_array(windows,NPY_FLOAT32,2,0) || IDIM(windows,1)!=6 || IDIM(windows,0)>64)) goto invalid;
    flightrl_inspection_render_sized(IDATA(p),IDATA(q),IDATA(r),IDATA(b),(int)IDIM(b,0),
                              IDATA(s),m,n,IDATA(f),IDATA(c),depth ? IDATA(depth) : NULL,(int)IDIM(f,2),(int)IDIM(f,1),materials,
        appearance ? IDATA(appearance):NULL,lights ? IDATA(lights):NULL,lights ? (int)IDIM(lights,0):0,windows ? IDATA(windows):NULL,windows ? (int)IDIM(windows,0):0);
    Py_RETURN_NONE;
invalid:
    PyErr_SetString(PyExc_ValueError,"inspection render requires finite contiguous typed scene/batch buffers");
    return NULL;
}
static PyObject *inspection_collision(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *a,*b,*r,*boxes,*c; float radius;
    if (!PyArg_ParseTuple(args,"OOOOfO",&a,&b,&r,&boxes,&radius,&c)) return NULL;
    if (!inspection_array(a,NPY_FLOAT32,2,0) || !inspection_array(b,NPY_FLOAT32,2,0) ||
        !inspection_array(r,NPY_FLOAT32,1,0) || !inspection_array(boxes,NPY_FLOAT32,2,0) ||
        !inspection_array(c,NPY_UINT8,1,1) || !isfinite(radius) || radius<=0) goto invalid;
    int n=(int)IDIM(a,0);
    if (n<1 || n>64 || IDIM(a,1)!=3 || IDIM(b,0)!=n || IDIM(b,1)!=3 ||
        IDIM(r,0)!=7 || IDIM(boxes,1)!=6 || IDIM(boxes,0)>1024 || IDIM(c,0)!=n) goto invalid;
    flightrl_inspection_collision(IDATA(a),IDATA(b),IDATA(r),IDATA(boxes),
                                 (int)IDIM(boxes,0),n,radius,IDATA(c));
    Py_RETURN_NONE;
invalid:
    PyErr_SetString(PyExc_ValueError,"inspection collision requires finite contiguous typed scene/batch buffers");
    return NULL;
}
#undef IDIM
#undef IDATA
#endif
