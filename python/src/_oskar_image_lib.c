/*
 * Copyright (c) 2014, The University of Oxford
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

#include <stdio.h>
#include <math.h>
#include <string.h>

#include <oskar_global.h>
#include <oskar_mem.h>
#include <oskar_image.h>
#include <oskar_evaluate_image_lm_grid.h>
#include <oskar_make_image_dft.h>

// http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static PyObject* OskarError;

typedef struct {
    double re;
    double im;
} dComplex;

static void make_image_dft_(double* image, int num_vis, const double* uu,
    const double* vv, const dComplex* amp, double freq, int size, double fov)
{
    int err = OSKAR_SUCCESS;
    int location = OSKAR_CPU;
    int num_pixels = size * size;
    int type = OSKAR_DOUBLE; // Note Also allow OSKAR_SINGLE?

    oskar_Mem *uu_, *vv_, *amp_;
    uu_ = oskar_mem_create(type, location, num_vis, &err);
    vv_ = oskar_mem_create(type, location, num_vis, &err);
    amp_ = oskar_mem_create(type | OSKAR_COMPLEX, location, num_vis, &err);
    memcpy(oskar_mem_double(uu_, &err), uu, num_vis*sizeof(double));
    memcpy(oskar_mem_double(vv_, &err), vv, num_vis*sizeof(double));
    memcpy(oskar_mem_double2(amp_, &err), amp, num_vis*sizeof(dComplex));

    oskar_Image* image_ = NULL;
    image_ = oskar_image_create(type, location, &err);
    oskar_image_resize(image_, size, size, 1, 1, 1, &err);
    oskar_image_set_centre(image_, 0.0, 0.0);
    oskar_image_set_fov(image_, fov*(180.0/M_PI), fov*(180.0/M_PI));
    oskar_image_set_time(image_, 0.0, 0.0);
    oskar_image_set_freq(image_, 0.0, 0.0);
    oskar_image_set_type(image_, 0);

    oskar_Mem *l_, *m_;
    l_ = oskar_mem_create(type, location, num_pixels, &err);
    m_ = oskar_mem_create(type, location, num_pixels, &err);

    oskar_evaluate_image_lm_grid_d(size, size, fov, fov,
        oskar_mem_double(l_, &err), oskar_mem_double(m_, &err));

    oskar_make_image_dft(oskar_image_data(image_), uu_, vv_, amp_, l_, m_, freq, &err);

    double* image_ptr_ = oskar_mem_double(oskar_image_data(image_), &err);
    memcpy(image, image_ptr_, num_pixels*sizeof(double));

    oskar_mem_free(uu_, &err);
    oskar_mem_free(vv_, &err);
    oskar_mem_free(amp_, &err);
    oskar_mem_free(l_, &err);
    oskar_mem_free(m_, &err);

    //printf("status = %i\n", err);
}

// TODO avoid memory duplication between Python and oskar_Mem/Image arrays.
static PyObject* make_image_dft(PyObject* self, PyObject* args, PyObject* keywds)
{
    // Function arguments.
    PyObject* uu_ = NULL;
    PyObject* vv_ = NULL;
    PyObject* amp_ = NULL;
    int size = 128;
    double fov_deg = 2.0;
    double freq;

    static char* keywords[] = {"uu", "vv", "amp", "freq", "fov", "size", NULL};
    //static char* keywords[] = {"fov", "size", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOd|di", keywords, &uu_,
        &vv_, &amp_, &freq, &fov_deg, &size))
        return NULL;

    // Parse inputs
    if (PyArray_NDIM((PyArrayObject*)uu_) != 1 ||
            PyArray_NDIM((PyArrayObject*)vv_) != 1 ||
            PyArray_NDIM((PyArrayObject*)amp_) != 1) {
        PyErr_SetString(OskarError, "Input data arrays must be 1D.");
        return NULL;
    }
    int num_vis = (int)*PyArray_DIMS((PyArrayObject*)uu_);
    if (num_vis != (int)*PyArray_DIMS((PyArrayObject*)vv_) ||
            num_vis != (int)*PyArray_DIMS((PyArrayObject*)amp_)) {
        PyErr_SetString(OskarError, "Input data dimension mismatch.");
        return NULL;
    }

    // Extract C arrays from Python objects.
    double* uu    = (double*)PyArray_DATA((PyArrayObject*)uu_);
    double* vv    = (double*)PyArray_DATA((PyArrayObject*)vv_);
    dComplex* amp = (dComplex*)PyArray_DATA((PyArrayObject*)amp_);

//    printf("Freq = %f\n", freq);
//    printf("FoV  = %f\n", fov_deg);
//    printf("size = %i\n", size);
//    for (int i = 0; i < 10; ++i) {
//        printf("==> %02i % -6.1f % -6.1f % -7.2e % -7.2e\n", i, uu[i], vv[i],
//                amp[i].re, amp[i].im);
//    }

    // Create memory for image data.
    npy_intp dims = size*size;
    PyArrayObject* image_ = (PyArrayObject*)PyArray_SimpleNew(1,&dims,NPY_DOUBLE);
    double* image = (double*)PyArray_DATA(image_);

    // Call wrapper function to make image.
    make_image_dft_(image, num_vis, uu, vv, amp, freq, size, fov_deg*(M_PI/180.0));

    // Return image to the python workspace.
    return Py_BuildValue("O", image_);
}

// Method table.
static PyMethodDef oskar_image_lib_methods[] =
{
        {"make", (PyCFunction)make_image_dft, METH_VARARGS | METH_KEYWORDS,
        "make_image(uu,vv,amp,freq,fov=2.0,size=128)\n\n"
        "Makes an image from visibility data. Computation is performed using a\n"
        "DFT implemented on the GPU using CUDA.\n\n"
        "Parameters\n"
        "----------\n"
        "uu : array like, shape (n,)\n"
        "   Input baseline uu coordinates, in metres.\n\n"
        "vv : array like, shape (n,)\n"
        "   Input baseline vv coordinates, in metres.\n\n"
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

// Initialisation function (called init[filename] where filename = name of *.so)
// http://docs.python.org/2/extending/extending.html
PyMODINIT_FUNC init_image_lib(void)
{
    PyObject* m = NULL;
    m = Py_InitModule3("_image_lib", oskar_image_lib_methods, "TODO: docstring...");
    if (m == NULL)
        return;
    OskarError = PyErr_NewException("oskar.error", NULL, NULL);
    Py_INCREF(OskarError);
    PyModule_AddObject(m, "error", OskarError);

    // Import the use of numpy array objects.
    import_array();
}
