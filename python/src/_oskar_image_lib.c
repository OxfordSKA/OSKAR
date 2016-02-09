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

#include <oskar_imager.h>
#include <oskar_get_error_string.h>
#include <string.h>

/* http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static PyObject* OskarError;

static PyObject* make_image(PyObject* self, PyObject* args, PyObject* keywds)
{
    PyArrayObject *uu = 0, *vv = 0, *ww = 0, *amp = 0, *im = 0;
    int i = 0, err = 0, num_vis, num_pixels, size = 0, type = OSKAR_DOUBLE;
    double fov = 0.0, norm = 0.0;
    oskar_Imager* imager;
    oskar_Mem *uu_c, *vv_c, *ww_c, *amp_c, *plane;

    /* Parse inputs. */
    char* keywords[] = {"uu", "vv", "ww", "amp", "fov", "size", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOdi", keywords,
            &uu, &vv, &ww, &amp, &fov, &size))
        return NULL;

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
    npy_intp dims[] = {size, size};
    im = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    /* Create and set up the imager. */
    imager = oskar_imager_create(type, &err);
    oskar_imager_set_fov(imager, fov);
    oskar_imager_set_size(imager, size);

    /* Pointers to input/output arrays. */
    uu_c = oskar_mem_create_alias_from_raw(PyArray_DATA(uu), type, OSKAR_CPU,
            num_vis, &err);
    vv_c = oskar_mem_create_alias_from_raw(PyArray_DATA(vv), type, OSKAR_CPU,
            num_vis, &err);
    ww_c = oskar_mem_create_alias_from_raw(PyArray_DATA(ww), type, OSKAR_CPU,
            num_vis, &err);
    amp_c = oskar_mem_create_alias_from_raw(PyArray_DATA(amp), type, OSKAR_CPU,
            num_vis, &err);

    /* Make the image. */
    plane = oskar_mem_create(type | OSKAR_COMPLEX, OSKAR_CPU, num_pixels, &err);
    oskar_imager_update_plane(imager, num_vis, uu_c, vv_c, ww_c, amp_c,
            plane, &norm, &err);
    oskar_imager_finalise_plane(imager, plane, norm, &err);
    memcpy(PyArray_DATA(im), oskar_mem_void_const(plane),
            num_pixels * sizeof(double));
    if (err)
        fprintf(stderr, "Error in imager: code %d (%s)\n",
                err, oskar_get_error_string(err));

    /* Free memory. */
    oskar_mem_free(uu_c, &err);
    oskar_mem_free(vv_c, &err);
    oskar_mem_free(ww_c, &err);
    oskar_mem_free(amp_c, &err);
    oskar_imager_free(imager, &err);

    /* Return image to the python workspace. */
    return Py_BuildValue("O", im);
}

/* Method table. */
static PyMethodDef oskar_image_lib_methods[] =
{
    {"make", (PyCFunction)make_image, METH_VARARGS | METH_KEYWORDS,
            "make(uu, vv, ww, amp, fov, size)\n\n"
            "Makes an image from visibility data.\n\n"
            "Parameters\n"
            "----------\n"
            "uu : array like, shape (n,)\n"
            "   Input baseline u coordinates, in wavelengths.\n\n"
            "vv : array like, shape (n,)\n"
            "   Input baseline v coordinates, in wavelengths.\n\n"
            "ww : array like, shape (n,)\n"
            "   Input baseline w coordinates, in wavelengths.\n\n"
            "amp : array like, shape (n,), complex\n"
            "   Input baseline amplitudes.\n\n"
            "fov : scalar\n"
            "   Image field of view, in degrees.\n\n"
            "size : integer\n"
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
