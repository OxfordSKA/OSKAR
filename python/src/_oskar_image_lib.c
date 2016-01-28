/*
 * Copyright (c) 2014-2016, The University of Oxford
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

#include <oskar_cmath.h>
#include <oskar_cuda_check_error.h>
#include <oskar_dft_c2r_3d_cuda.h>
#include <oskar_evaluate_image_lmn_grid.h>
#include <oskar_mem.h>

/* http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static PyObject* OskarError;

typedef struct {
    double re;
    double im;
} dComplex;

static PyObject* make_image_dft(PyObject* self, PyObject* args, PyObject* keywds)
{
    PyArrayObject *uu = 0, *vv = 0, *ww = 0, *amp = 0, *im = 0;
    int size = 128, err = 0;
    double fov = 2.0, freq, wavenumber;
    int i, num_vis, num_pixels, type = OSKAR_DOUBLE;
    oskar_Mem *uu_c, *vv_c, *ww_c, *amp_c, *l_c, *m_c, *n_c, *im_c;
    oskar_Mem *uu_g, *vv_g, *ww_g, *amp_g, *l_g, *m_g, *n_g, *im_g;

    /* Parse inputs. */
    char* keywords[] = {"uu", "vv", "ww", "amp", "freq", "fov", "size",
            NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOd|di", keywords,
            &uu, &vv, &ww, &amp, &freq, &fov, &size))
        return NULL;
    fov *= (M_PI / 180.0);

    /* Check dimensions. */
    if (PyArray_NDIM(uu) != 1 || PyArray_NDIM(vv) != 1 ||
            PyArray_NDIM(ww) != 1 || PyArray_NDIM(amp) != 1)
    {
        PyErr_SetString(OskarError, "Input data arrays must be 1D.");
        return NULL;
    }
    num_vis = (int)*PyArray_DIMS(uu);
    if (num_vis != (int)*PyArray_DIMS(vv) ||
            num_vis != (int)*PyArray_DIMS(ww) ||
            num_vis != (int)*PyArray_DIMS(amp))
    {
        PyErr_SetString(OskarError, "Input data dimension mismatch.");
        return NULL;
    }

    /* Create the output image array. */
    num_pixels = size * size;
    npy_intp dims = num_pixels;
    im = (PyArrayObject*)PyArray_SimpleNew(1, &dims, NPY_DOUBLE);

    /* Pointers to input/output arrays. */
    uu_c = oskar_mem_create_alias_from_raw(PyArray_DATA(uu), type, OSKAR_CPU,
            num_vis, &err);
    vv_c = oskar_mem_create_alias_from_raw(PyArray_DATA(vv), type, OSKAR_CPU,
            num_vis, &err);
    ww_c = oskar_mem_create_alias_from_raw(PyArray_DATA(ww), type, OSKAR_CPU,
            num_vis, &err);
    im_c = oskar_mem_create_alias_from_raw(PyArray_DATA(im), type, OSKAR_CPU,
            num_pixels, &err);
    amp_c = oskar_mem_create_alias_from_raw(PyArray_DATA(amp), type, OSKAR_CPU,
            num_vis, &err);

    /* Create the image grid. */
    l_c = oskar_mem_create(type, OSKAR_CPU, num_pixels, &err);
    m_c = oskar_mem_create(type, OSKAR_CPU, num_pixels, &err);
    n_c = oskar_mem_create(type, OSKAR_CPU, num_pixels, &err);
    oskar_evaluate_image_lmn_grid(size, size, fov, fov, 0, l_c, m_c, n_c, &err);
    oskar_mem_add_real(n_c, -1.0, &err);

    /* Copy input data to GPU. */
    uu_g = oskar_mem_create_copy(uu_c, OSKAR_GPU, &err);
    vv_g = oskar_mem_create_copy(vv_c, OSKAR_GPU, &err);
    ww_g = oskar_mem_create_copy(ww_c, OSKAR_GPU, &err);
    amp_g = oskar_mem_create_copy(amp_c, OSKAR_GPU, &err);
    l_g = oskar_mem_create_copy(l_c, OSKAR_GPU, &err);
    m_g = oskar_mem_create_copy(m_c, OSKAR_GPU, &err);
    n_g = oskar_mem_create_copy(n_c, OSKAR_GPU, &err);

    /* Make the image. */
    im_g = oskar_mem_create(type, OSKAR_GPU, num_pixels, &err);
    wavenumber = 2.0 * M_PI * freq / 299792458.0;
    if (!err)
    {
        if (type == OSKAR_SINGLE)
            oskar_dft_c2r_3d_cuda_f(num_vis, wavenumber,
                    oskar_mem_float_const(uu_g, &err),
                    oskar_mem_float_const(vv_g, &err),
                    oskar_mem_float_const(ww_g, &err),
                    oskar_mem_float2_const(amp_g, &err), num_pixels,
                    oskar_mem_float_const(l_g, &err),
                    oskar_mem_float_const(m_g, &err),
                    oskar_mem_float_const(n_g, &err),
                    oskar_mem_float(im_g, &err));
        else
            oskar_dft_c2r_3d_cuda_d(num_vis, wavenumber,
                    oskar_mem_double_const(uu_g, &err),
                    oskar_mem_double_const(vv_g, &err),
                    oskar_mem_double_const(ww_g, &err),
                    oskar_mem_double2_const(amp_g, &err), num_pixels,
                    oskar_mem_double_const(l_g, &err),
                    oskar_mem_double_const(m_g, &err),
                    oskar_mem_double_const(n_g, &err),
                    oskar_mem_double(im_g, &err));
    }
    oskar_cuda_check_error(&err);
    oskar_mem_scale_real(im_g, 1.0 / num_vis, &err);

    /* Copy image data back from GPU. */
    oskar_mem_copy(im_c, im_g, &err);

    /* Free memory. */
    oskar_mem_free(uu_c, &err);
    oskar_mem_free(uu_g, &err);
    oskar_mem_free(vv_c, &err);
    oskar_mem_free(vv_g, &err);
    oskar_mem_free(ww_c, &err);
    oskar_mem_free(ww_g, &err);
    oskar_mem_free(amp_c, &err);
    oskar_mem_free(amp_g, &err);
    oskar_mem_free(l_c, &err);
    oskar_mem_free(l_g, &err);
    oskar_mem_free(m_c, &err);
    oskar_mem_free(m_g, &err);
    oskar_mem_free(n_c, &err);
    oskar_mem_free(n_g, &err);
    oskar_mem_free(im_c, &err);
    oskar_mem_free(im_g, &err);

    /* Return image to the python workspace. */
    return Py_BuildValue("O", im);
}

/* Method table. */
static PyMethodDef oskar_image_lib_methods[] =
{
    {"make", (PyCFunction)make_image_dft, METH_VARARGS | METH_KEYWORDS,
            "make(uu, vv, ww, amp, freq, fov=2.0, size=128)\n\n"
            "Makes an image from visibility data using a DFT.\n\n"
            "Parameters\n"
            "----------\n"
            "uu : array like, shape (n,)\n"
            "   Input baseline u coordinates, in metres.\n\n"
            "vv : array like, shape (n,)\n"
            "   Input baseline v coordinates, in metres.\n\n"
            "ww : array like, shape (n,)\n"
            "   Input baseline w coordinates, in metres.\n\n"
            "amp : array like, shape (n,), complex\n"
            "   Input baseline amplitudes.\n\n"
            "freq : scalar\n"
            "   Frequency, in Hz.\n\n"
            "fov : scalar, default=2.0\n"
            "   Image field of view, in degrees.\n\n"
            "size : integer, default=128\n"
            "   Image size along one dimension, in pixels.\n\n"
    },
    {NULL, NULL, 0, NULL}
};

/* Initialisation function (called init[filename] where filename = name of *.so)
 * http://docs.python.org/2/extending/extending.html */
PyMODINIT_FUNC init_image_lib(void)
{
    PyObject* m = NULL;
    m = Py_InitModule3("_image_lib", oskar_image_lib_methods, "TODO: docstring...");
    if (m == NULL)
        return;
    OskarError = PyErr_NewException("oskar.error", NULL, NULL);
    Py_INCREF(OskarError);
    PyModule_AddObject(m, "error", OskarError);

    /* Import the use of numpy array objects. */
    import_array();
}
