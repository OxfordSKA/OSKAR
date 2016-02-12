/*
 * Copyright (c) 2016, The University of Oxford
 * All rights reserved.
 *
 * This file is part of the OSKAR package.
 * Contact: oskar at oerc.ox.ac.uk
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <Python.h>

/* http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html */
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stddef.h>

typedef struct DComplex {
    double re;
    double im;
} DComplex;

/**
 * @brief
 * @details
 */
static PyObject* apply_gains(PyObject* self, PyObject* args)
{
    /* Read input arguments */
    PyObject *vis_in_ = NULL;
    PyObject *gains_ = NULL;
    if (!PyArg_ParseTuple(args, "OO", &vis_in_, &gains_))
        return NULL;

    /* Return an ndarray from the python objects */
    int typenum = NPY_CDOUBLE;
    int requirements = NPY_ARRAY_IN_ARRAY;
    PyObject* pyo_vis_in = PyArray_FROM_OTF(vis_in_, typenum, requirements);
    if (!pyo_vis_in) goto fail;
    PyObject* pyo_gains = PyArray_FROM_OTF(gains_, typenum, requirements);
    if (!pyo_gains) goto fail;

    /* TODO(BM) Error checking on the dims! */
    int nd_vis = PyArray_NDIM((PyArrayObject*)pyo_vis_in);
    npy_intp* vis_dims = PyArray_DIMS((PyArrayObject*)pyo_vis_in);
    int num_vis = vis_dims[0];

    int nd_gains = PyArray_NDIM((PyArrayObject*)pyo_gains);
    npy_intp* gains_dims = PyArray_DIMS((PyArrayObject*)pyo_gains);
    int num_antennas = gains_dims[0];

    /* Create PyObject for output visibilities */
    PyObject* pyo_vis_out = PyArray_SimpleNew(nd_vis, vis_dims, NPY_CDOUBLE);

    /* Apply the gains: v_out = gp * v_in * conj(gq) */
    DComplex* v_out = (DComplex*)PyArray_DATA((PyArrayObject*)pyo_vis_out);
    DComplex* g = (DComplex*)PyArray_DATA((PyArrayObject*)pyo_gains);
    DComplex* v_in = (DComplex*)PyArray_DATA((PyArrayObject*)pyo_vis_in);
    for (int i = 0, p = 0; p < num_antennas; ++p) {
        for (int q = p + 1; q < num_antennas; ++q, ++i) {
            double a = v_in[i].re * g[p].re - v_in[i].im * g[p].im;
            double b = v_in[i].im * g[p].re + v_in[i].re * g[p].im;
            v_out[i].re = a * g[q].re + b * g[q].im;
            v_out[i].im = b * g[q].re - a * g[q].im;
        }
    }
    /* Decrement references to temporary array objects. */
    Py_DECREF(pyo_vis_in);
    Py_DECREF(pyo_gains);
    return Py_BuildValue("N", pyo_vis_out);

    printf("  - Ref count: %zi, %zi, %zi\n",
            PyArray_REFCOUNT(pyo_vis_in),
            PyArray_REFCOUNT(pyo_gains),
            PyArray_REFCOUNT(pyo_vis_out));

fail:
    Py_XDECREF(pyo_gains);
    Py_XDECREF(pyo_vis_in);
    return NULL;
}


/**
 * @brief
 * @details
 */
static PyObject* vis_list_to_matrix(PyObject* self, PyObject* args)
{
    /* Read input arguments */
    PyObject* vis_list_ = NULL;
    int num_ant = 0;
    if (!PyArg_ParseTuple(args, "Oi", &vis_list_, &num_ant))
        return NULL;

    /* Convert to an ndarray */
    PyObject* pyo_vis_list = PyArray_FROM_OTF(vis_list_, NPY_CDOUBLE,
                                              NPY_ARRAY_IN_ARRAY);
    if (!pyo_vis_list) goto fail;

    /* TODO(BM) Error checking on the dims! */
    int nd_vis = PyArray_NDIM((PyArrayObject*)pyo_vis_list);
    npy_intp* vis_dims = PyArray_DIMS((PyArrayObject*)pyo_vis_list);
    int num_vis = vis_dims[0];

    /* Create PyObject for output visibilities */
    npy_intp dims[] = { num_ant, num_ant };
    PyObject* pyo_vis_matrix = PyArray_SimpleNew(2, dims, NPY_CDOUBLE);

    DComplex* v_list = (DComplex*)PyArray_DATA((PyArrayObject*)pyo_vis_list);
    DComplex* v_matrix = (DComplex*)PyArray_DATA((PyArrayObject*)pyo_vis_matrix);
    for (int i = 0, p = 0; p < num_ant; ++p) {
        for (int q = p + 1; q < num_ant; ++q, ++i) {
            v_matrix[p * num_ant + q].re = v_list[i].re;
            v_matrix[p * num_ant + q].im = -v_list[i].im;
            v_matrix[q * num_ant + p].re = v_list[i].re;
            v_matrix[q * num_ant + p].im = v_list[i].im;
        }
    }
    for (int i = 0; i < num_ant; ++i) {
        v_matrix[i * num_ant + i].re = 0.0;
        v_matrix[i * num_ant + i].im = 0.0;
    }

    /* Decrement references to temporary array objects. */
    Py_DECREF(pyo_vis_list);
    return Py_BuildValue("N", pyo_vis_matrix);

fail:
    Py_XDECREF(pyo_vis_list);
    return NULL;
}

static PyObject* check_ref_count(PyObject* self, PyObject* args)
{
    PyObject* obj = NULL;
    /* https://docs.python.org/2/c-api/arg.html */
    /* Reference count is not increased by 'O' for PyArg_ParseTyple */
    if (!PyArg_ParseTuple(args, "O", &obj))
        return NULL;
    return Py_BuildValue("ii", PyArray_REFCOUNT(obj), Py_REFCNT(obj));
}

/* Method table. */
static PyMethodDef methods[] =
{
    {
        "apply_gains",
        (PyCFunction)apply_gains, METH_VARARGS,
        "vis_amp = apply_gains(gains, vis_amp)\n"
        "Applies gains.\n"
    },
    {
        "vis_list_to_matrix",
        (PyCFunction)vis_list_to_matrix, METH_VARARGS,
        "vis_matrix = vis_list_to_matrix(vis_list)\n"
        "Converts a list of visibilities to matrix form.\n"
    },
    {
        "check_ref_count",
        (PyCFunction)check_ref_count, METH_VARARGS,
        "count = check_ref_count(PyObject)\n"
        "Check the reference count of a python object\n"
    },
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_bda_utils",       /* m_name */
        NULL,               /* m_doc */
        -1,                 /* m_size */
        methods             /* m_methods */
};
#endif


static PyObject* moduleinit(void)
{
    PyObject* m;
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3("_bda_utils", methods, "docstring ...");
#endif
    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__bda_utils(void)
{
    import_array();
    return moduleinit();
}
#else
// XXX the init function name has to match that of the compiled module
// with the pattern 'init<module name>'. This module is called '_apply_gains'
void init_bda_utils(void)
{
    import_array();
    moduleinit();
    return;
}
#endif


