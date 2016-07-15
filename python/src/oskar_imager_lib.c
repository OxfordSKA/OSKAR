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
#include <oskar_vis_block.h>
#include <oskar_get_error_string.h>
#include <string.h>

/* http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static const char* module_doc =
        "This module provides an interface to the OSKAR imager.";
static const char* name = "oskar_Imager";

static void imager_free(PyObject* capsule)
{
    int status = 0;
    oskar_Imager* h = (oskar_Imager*) PyCapsule_GetPointer(capsule, name);
    oskar_imager_free(h, &status);
}


static oskar_Imager* get_handle_imager(PyObject* capsule)
{
    oskar_Imager* h = 0;
    if (!PyCapsule_CheckExact(capsule))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input is not a PyCapsule object!");
        return 0;
    }
    h = (oskar_Imager*) PyCapsule_GetPointer(capsule, name);
    if (!h)
    {
        PyErr_SetString(PyExc_RuntimeError,
                "Unable to convert PyCapsule object to oskar_Imager.");
        return 0;
    }
    return h;
}


static oskar_VisBlock* get_handle_vis_block(PyObject* capsule)
{
    oskar_VisBlock* h = 0;
    if (!PyCapsule_CheckExact(capsule))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input is not a PyCapsule object!");
        return 0;
    }
    h = (oskar_VisBlock*) PyCapsule_GetPointer(capsule, "oskar_VisBlock");
    if (!h)
    {
        PyErr_SetString(PyExc_RuntimeError,
                "Unable to convert PyCapsule object to oskar_VisBlock.");
        return 0;
    }
    return h;
}


static oskar_VisHeader* get_handle_vis_header(PyObject* capsule)
{
    oskar_VisHeader* h = 0;
    if (!PyCapsule_CheckExact(capsule))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input is not a PyCapsule object!");
        return 0;
    }
    h = (oskar_VisHeader*) PyCapsule_GetPointer(capsule, "oskar_VisHeader");
    if (!h)
    {
        PyErr_SetString(PyExc_RuntimeError,
                "Unable to convert PyCapsule object to oskar_VisHeader.");
        return 0;
    }
    return h;
}


static int oskar_type_from_numpy(PyArrayObject* arr)
{
    switch (PyArray_TYPE(arr))
    {
    case NPY_INT:     return OSKAR_INT;
    case NPY_FLOAT:   return OSKAR_SINGLE;
    case NPY_DOUBLE:  return OSKAR_DOUBLE;
    case NPY_CFLOAT:  return OSKAR_SINGLE_COMPLEX;
    case NPY_CDOUBLE: return OSKAR_DOUBLE_COMPLEX;
    }
    return 0;
}


static PyObject* algorithm(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    return Py_BuildValue("s", oskar_imager_algorithm(h));
}


static PyObject* cellsize(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    return Py_BuildValue("d", oskar_imager_cellsize(h));
}


static PyObject* channel_end(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    return Py_BuildValue("i", oskar_imager_channel_end(h));
}


static PyObject* channel_snapshots(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int flag = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    flag = oskar_imager_channel_snapshots(h);
    return Py_BuildValue("O", flag ? Py_True : Py_False);
}


static PyObject* channel_start(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    return Py_BuildValue("i", oskar_imager_channel_start(h));
}


static PyObject* check_init(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    Py_BEGIN_ALLOW_THREADS
    oskar_imager_check_init(h, &status);
    Py_END_ALLOW_THREADS
    return Py_BuildValue("i", status);
}


static PyObject* coords_only(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int flag;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    flag = oskar_imager_coords_only(h);
    return Py_BuildValue("O", flag ? Py_True : Py_False);
}


static PyObject* create(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0, prec = 0;
    const char* type;
    if (!PyArg_ParseTuple(args, "s", &type)) return 0;
    prec = (type[0] == 'S' || type[0] == 's') ? OSKAR_SINGLE : OSKAR_DOUBLE;
    h = oskar_imager_create(prec, &status);
    capsule = PyCapsule_New((void*)h, name, (PyCapsule_Destructor)imager_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* finalise(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject *obj[] = {0, 0};
    PyArrayObject* plane = 0;
    oskar_Mem* plane_c = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "OO", &obj[0], &obj[1])) return 0;
    if (!(h = get_handle_imager(obj[0]))) return 0;

    /* Check if an output image plane was given. */
    if (obj[1] != Py_None)
    {
        plane = (PyArrayObject*) PyArray_FROM_OF(obj[1], NPY_ARRAY_OUT_ARRAY);
        if (plane)
        {
            plane_c = oskar_mem_create_alias_from_raw(PyArray_DATA(plane),
                    oskar_type_from_numpy(plane), OSKAR_CPU,
                    PyArray_SIZE(plane), &status);
        }
    }

    /* Finalise. */
    Py_BEGIN_ALLOW_THREADS
    oskar_imager_finalise(h, plane_c, &status);
    Py_END_ALLOW_THREADS
    oskar_mem_free(plane_c, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_finalise() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        goto fail;
    }
    Py_XDECREF(plane);
    return Py_BuildValue("");

fail:
    Py_XDECREF(plane);
    return 0;
}


static PyObject* finalise_plane(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject *obj[] = {0, 0};
    PyArrayObject* plane = 0;
    oskar_Mem* plane_c = 0;
    int status = 0;
    double plane_norm = 0.0;
    if (!PyArg_ParseTuple(args, "OOd", &obj[0], &obj[1], &plane_norm))
        return 0;
    if (!(h = get_handle_imager(obj[0]))) return 0;

    /* Get the supplied plane. */
    plane = (PyArrayObject*) PyArray_FROM_OF(obj[1], NPY_ARRAY_OUT_ARRAY);
    if (!plane) goto fail;
    plane_c = oskar_mem_create_alias_from_raw(PyArray_DATA(plane),
            oskar_type_from_numpy(plane), OSKAR_CPU,
            PyArray_SIZE(plane), &status);

    /* Finalise the plane. */
    Py_BEGIN_ALLOW_THREADS
    oskar_imager_finalise_plane(h, plane_c, plane_norm, &status);
    Py_END_ALLOW_THREADS
    oskar_mem_free(plane_c, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_finalise_plane() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        goto fail;
    }
    Py_XDECREF(plane);
    return Py_BuildValue("");

fail:
    Py_XDECREF(plane);
    return 0;
}


static PyObject* fov(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    return Py_BuildValue("d", oskar_imager_fov(h));
}


static PyObject* image_size(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    return Py_BuildValue("i", oskar_imager_image_size(h));
}


static PyObject* image_type(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    return Py_BuildValue("s", oskar_imager_image_type(h));
}


static PyObject* input_file(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    return Py_BuildValue("s", oskar_imager_input_file(h));
}


static PyObject* ms_column(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    return Py_BuildValue("s", oskar_imager_ms_column(h));
}


static PyObject* num_w_planes(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    return Py_BuildValue("i", oskar_imager_num_w_planes(h));
}


static PyObject* output_root(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    return Py_BuildValue("s", oskar_imager_output_root(h));
}


static PyObject* plane_size(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    return Py_BuildValue("i", oskar_imager_plane_size(h));
}


static PyObject* reset_cache(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_reset_cache(h, &status);
    return Py_BuildValue("i", status);
}


static PyObject* run(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    Py_BEGIN_ALLOW_THREADS
    oskar_imager_run(h, &status);
    Py_END_ALLOW_THREADS

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_run() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* set_algorithm(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* type = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &type)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_algorithm(h, type, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_set_algorithm() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* set_cellsize(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    double cellsize = 0.0;
    if (!PyArg_ParseTuple(args, "Od", &capsule, &cellsize)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_cellsize(h, cellsize);
    return Py_BuildValue("");
}


static PyObject* set_channel_end(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_channel_end(h, value);
    return Py_BuildValue("");
}


static PyObject* set_channel_snapshots(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_channel_snapshots(h, value);
    return Py_BuildValue("");
}


static PyObject* set_channel_start(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_channel_start(h, value);
    return Py_BuildValue("");
}


static PyObject* set_coords_only(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int flag = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &flag)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_coords_only(h, flag);
    return Py_BuildValue("");
}


static PyObject* set_default_direction(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_default_direction(h);
    return Py_BuildValue("");
}


static PyObject* set_direction(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    double ra = 0.0, dec = 0.0;
    if (!PyArg_ParseTuple(args, "Odd", &capsule, &ra, &dec)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_direction(h, ra, dec);
    return Py_BuildValue("");
}


static PyObject* set_fft_on_gpu(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_fft_on_gpu(h, value);
    return Py_BuildValue("");
}


static PyObject* set_fov(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    double fov = 0.0;
    if (!PyArg_ParseTuple(args, "Od", &capsule, &fov)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_fov(h, fov);
    return Py_BuildValue("");
}


static PyObject* set_grid_kernel(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0, support = 0, oversample = 0;
    const char* type = 0;
    if (!PyArg_ParseTuple(args, "Osii", &capsule, &type, &support, &oversample))
        return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_grid_kernel(h, type, support, oversample, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_set_grid_kernel() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("i", status);
}


static PyObject* set_image_size(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int size = 0, status = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &size)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_image_size(h, size, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_set_image_size() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* set_image_type(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* type = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &type)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_image_type(h, type, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_set_image_type() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("i", status);
}


static PyObject* set_input_file(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* filename = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &filename)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_input_file(h, filename, &status);
    return Py_BuildValue("i", status);
}


static PyObject* set_ms_column(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* column = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &column)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_ms_column(h, column, &status);
    return Py_BuildValue("i", status);
}


static PyObject* set_num_w_planes(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int num = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &num)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_num_w_planes(h, num);
    return Py_BuildValue("");
}


static PyObject* set_output_root(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* filename = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &filename)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_output_root(h, filename, &status);
    return Py_BuildValue("i", status);
}


static PyObject* set_size(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int size = 0, status = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &size)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_size(h, size, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_set_size() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* set_time_end(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_time_end(h, value);
    return Py_BuildValue("");
}


static PyObject* set_time_snapshots(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_time_snapshots(h, value);
    return Py_BuildValue("");
}


static PyObject* set_time_start(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_time_start(h, value);
    return Py_BuildValue("");
}


static PyObject* set_vis_frequency(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0, num = 0;
    double ref = 0.0, inc = 0.0;
    if (!PyArg_ParseTuple(args, "Oddi", &capsule, &ref, &inc, &num)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_vis_frequency(h, ref, inc, num, &status);
    return Py_BuildValue("i", status);
}


static PyObject* set_vis_phase_centre(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    double ra = 0.0, dec = 0.0;
    if (!PyArg_ParseTuple(args, "Odd", &capsule, &ra, &dec)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_vis_phase_centre(h, ra, dec);
    return Py_BuildValue("");
}


static PyObject* set_vis_time(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0, num = 0;
    double ref = 0.0, inc = 0.0;
    if (!PyArg_ParseTuple(args, "Oddi", &capsule, &ref, &inc, &num)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_vis_time(h, ref, inc, num, &status);
    return Py_BuildValue("i", status);
}


static PyObject* set_weighting(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* type = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &type)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    oskar_imager_set_weighting(h, type, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_set_weighting() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* size(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    return Py_BuildValue("i", oskar_imager_size(h));
}


static PyObject* time_end(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    return Py_BuildValue("i", oskar_imager_time_end(h));
}


static PyObject* time_snapshots(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int flag = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    flag = oskar_imager_time_snapshots(h);
    return Py_BuildValue("O", flag ? Py_True : Py_False);
}


static PyObject* time_start(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    return Py_BuildValue("i", oskar_imager_time_start(h));
}


static PyObject* update(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject *obj[] = {0, 0, 0, 0, 0, 0};
    oskar_Mem *uu_c, *vv_c, *ww_c, *amp_c, *weight_c;
    PyArrayObject *uu = 0, *vv = 0, *ww = 0, *amps = 0, *weight = 0;
    int start_time = 0, end_time = 0, start_chan = 0, end_chan = 0;
    int num_pols = 1, num_baselines = 0, vis_type;
    int num_times, num_chan, num_coords, num_vis, num_weights, status = 0;

    /* Parse inputs. */
    if (!PyArg_ParseTuple(args, "OiOOOOOiiiii", &obj[0],
            &num_baselines, &obj[1], &obj[2], &obj[3], &obj[4], &obj[5],
            &num_pols, &start_time, &end_time, &start_chan, &end_chan))
        return 0;
    if (!(h = get_handle_imager(obj[0]))) return 0;

    /* Make sure input objects are arrays. Convert if required. */
    uu     = (PyArrayObject*) PyArray_FROM_OF(obj[1], NPY_ARRAY_IN_ARRAY);
    vv     = (PyArrayObject*) PyArray_FROM_OF(obj[2], NPY_ARRAY_IN_ARRAY);
    ww     = (PyArrayObject*) PyArray_FROM_OF(obj[3], NPY_ARRAY_IN_ARRAY);
    amps   = (PyArrayObject*) PyArray_FROM_OF(obj[4], NPY_ARRAY_IN_ARRAY);
    weight = (PyArrayObject*) PyArray_FROM_OF(obj[5], NPY_ARRAY_IN_ARRAY);
    if (!uu || !vv || !ww || !amps || !weight)
        goto fail;

    /* Check visibility data are complex. */
    if (!PyArray_ISCOMPLEX(amps))
    {
        PyErr_SetString(PyExc_RuntimeError,
                "Input visibility data must be complex.");
        goto fail;
    }

    /* Get dimensions. */
    num_times = 1 + end_time - start_time;
    num_chan = 1 + end_chan - start_chan;
    num_coords = num_times * num_baselines;
    num_vis = num_coords * num_chan;
    num_weights = num_coords * num_pols;
    vis_type = oskar_type_from_numpy(amps);
    if (num_pols == 4) vis_type |= OSKAR_MATRIX;

    /* Pointers to input arrays. */
    uu_c = oskar_mem_create_alias_from_raw(PyArray_DATA(uu),
            oskar_type_from_numpy(uu), OSKAR_CPU, num_coords, &status);
    vv_c = oskar_mem_create_alias_from_raw(PyArray_DATA(vv),
            oskar_type_from_numpy(vv), OSKAR_CPU, num_coords, &status);
    ww_c = oskar_mem_create_alias_from_raw(PyArray_DATA(ww),
            oskar_type_from_numpy(ww), OSKAR_CPU, num_coords, &status);
    amp_c = oskar_mem_create_alias_from_raw(PyArray_DATA(amps),
            vis_type, OSKAR_CPU, num_vis, &status);
    weight_c = oskar_mem_create_alias_from_raw(PyArray_DATA(weight),
            oskar_type_from_numpy(weight), OSKAR_CPU, num_weights, &status);

    /* Update the imager with the supplied visibility data. */
    Py_BEGIN_ALLOW_THREADS
    oskar_imager_update(h, start_time, end_time, start_chan, end_chan,
            num_pols, num_baselines, uu_c, vv_c, ww_c, amp_c, weight_c,
            &status);
    Py_END_ALLOW_THREADS
    oskar_mem_free(uu_c, &status);
    oskar_mem_free(vv_c, &status);
    oskar_mem_free(ww_c, &status);
    oskar_mem_free(amp_c, &status);
    oskar_mem_free(weight_c, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_update() failed with code %d (%s).", status,
                oskar_get_error_string(status));
        goto fail;
    }
    Py_XDECREF(uu);
    Py_XDECREF(vv);
    Py_XDECREF(ww);
    Py_XDECREF(amps);
    Py_XDECREF(weight);
    return Py_BuildValue("");

fail:
    Py_XDECREF(uu);
    Py_XDECREF(vv);
    Py_XDECREF(ww);
    Py_XDECREF(amps);
    Py_XDECREF(weight);
    return 0;
}


static PyObject* update_from_block(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    oskar_VisBlock* block = 0;
    oskar_VisHeader* header = 0;
    PyObject *obj[] = {0, 0, 0};
    int status = 0;

    /* Parse inputs. */
    if (!PyArg_ParseTuple(args, "OOO", &obj[0], &obj[1], &obj[2]))
        return 0;
    if (!(h = get_handle_imager(obj[0]))) return 0;
    if (!(header = get_handle_vis_header(obj[1]))) return 0;
    if (!(block = get_handle_vis_block(obj[2]))) return 0;

    /* Update the imager with the supplied visibility data. */
    Py_BEGIN_ALLOW_THREADS
    oskar_imager_update_from_block(h, header, block, &status);
    Py_END_ALLOW_THREADS

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_update_block() failed with code %d (%s).", status,
                oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* update_plane(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject *obj[] = {0, 0, 0, 0, 0, 0, 0, 0};
    oskar_Mem *uu_c = 0, *vv_c = 0, *ww_c = 0, *amp_c = 0, *weight_c = 0;
    oskar_Mem *plane_c = 0, *weights_grid_c = 0;
    PyArrayObject *uu = 0, *vv = 0, *ww = 0, *amps = 0, *weight = 0;
    PyArrayObject *plane = 0, *weights_grid = 0;
    double plane_norm = 0.0;
    int num_vis, status = 0;

    /* Parse inputs:
     * capsule, uu, vv, ww, amps, weight, plane, plane_norm, weights_grid. */
    if (!PyArg_ParseTuple(args, "OOOOOOOdO", &obj[0],
            &obj[1], &obj[2], &obj[3], &obj[4], &obj[5], &obj[6], &plane_norm,
            &obj[7]))
        return 0;
    if (!(h = get_handle_imager(obj[0]))) return 0;

    /* Make sure input objects are arrays. Convert if required. */
    uu     = (PyArrayObject*) PyArray_FROM_OF(obj[1], NPY_ARRAY_IN_ARRAY);
    vv     = (PyArrayObject*) PyArray_FROM_OF(obj[2], NPY_ARRAY_IN_ARRAY);
    ww     = (PyArrayObject*) PyArray_FROM_OF(obj[3], NPY_ARRAY_IN_ARRAY);
    amps   = (PyArrayObject*) PyArray_FROM_OF(obj[4], NPY_ARRAY_IN_ARRAY);
    weight = (PyArrayObject*) PyArray_FROM_OF(obj[5], NPY_ARRAY_IN_ARRAY);
    plane  = (PyArrayObject*) PyArray_FROM_OF(obj[6], NPY_ARRAY_OUT_ARRAY);
    if (!uu || !vv || !ww || !amps || !weight || !plane)
        goto fail;

    /* Check if weights grid is present. */
    if (obj[7] != Py_None)
    {
        weights_grid = (PyArrayObject*) PyArray_FROM_OF(obj[7],
                NPY_ARRAY_IN_ARRAY);
        if (!weights_grid) goto fail;
    }

    /* Check dimensions. */
    if (PyArray_NDIM(uu) != 1 || PyArray_NDIM(vv) != 1 ||
            PyArray_NDIM(ww) != 1 || PyArray_NDIM(amps) != 1 ||
            PyArray_NDIM(weight) != 1)
    {
        PyErr_SetString(PyExc_RuntimeError, "Input data arrays must be 1D.");
        goto fail;
    }
    if (PyArray_NDIM(plane) != 2)
    {
        PyErr_SetString(PyExc_RuntimeError, "Plane must be 2D.");
        goto fail;
    }
    num_vis = (int) PyArray_SIZE(amps);
    if (num_vis != (int) PyArray_SIZE(uu) ||
            num_vis != (int) PyArray_SIZE(vv) ||
            num_vis != (int) PyArray_SIZE(ww) ||
            num_vis != (int) PyArray_SIZE(weight))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input data dimension mismatch.");
        goto fail;
    }

    /* Check visibility data are complex. */
    if (!PyArray_ISCOMPLEX(amps))
    {
        PyErr_SetString(PyExc_RuntimeError,
                "Input visibility data must be complex.");
        goto fail;
    }

    /* Pointers to input arrays. */
    uu_c = oskar_mem_create_alias_from_raw(PyArray_DATA(uu),
            oskar_type_from_numpy(uu), OSKAR_CPU, num_vis, &status);
    vv_c = oskar_mem_create_alias_from_raw(PyArray_DATA(vv),
            oskar_type_from_numpy(vv), OSKAR_CPU, num_vis, &status);
    ww_c = oskar_mem_create_alias_from_raw(PyArray_DATA(ww),
            oskar_type_from_numpy(ww), OSKAR_CPU, num_vis, &status);
    amp_c = oskar_mem_create_alias_from_raw(PyArray_DATA(amps),
            oskar_type_from_numpy(amps), OSKAR_CPU, num_vis, &status);
    weight_c = oskar_mem_create_alias_from_raw(PyArray_DATA(weight),
            oskar_type_from_numpy(weight), OSKAR_CPU, num_vis, &status);
    plane_c = oskar_mem_create_alias_from_raw(PyArray_DATA(plane),
            oskar_type_from_numpy(plane), OSKAR_CPU,
            (size_t) PyArray_SIZE(plane), &status);
    if (weights_grid)
    {
        weights_grid_c = oskar_mem_create_alias_from_raw(
                PyArray_DATA(weights_grid),
                oskar_type_from_numpy(weights_grid), OSKAR_CPU,
                (size_t) PyArray_SIZE(weights_grid), &status);
    }

    /* Update the plane. */
    Py_BEGIN_ALLOW_THREADS
    oskar_imager_update_plane(h, num_vis, uu_c, vv_c, ww_c, amp_c,
            weight_c, plane_c, &plane_norm, weights_grid_c, &status);
    Py_END_ALLOW_THREADS

    /* Clean up. */
    oskar_mem_free(uu_c, &status);
    oskar_mem_free(vv_c, &status);
    oskar_mem_free(ww_c, &status);
    oskar_mem_free(amp_c, &status);
    oskar_mem_free(weight_c, &status);
    oskar_mem_free(plane_c, &status);
    oskar_mem_free(weights_grid_c, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_update_plane() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        goto fail;
    }
    Py_XDECREF(uu);
    Py_XDECREF(vv);
    Py_XDECREF(ww);
    Py_XDECREF(amps);
    Py_XDECREF(weight);
    Py_XDECREF(plane);
    Py_XDECREF(weights_grid);
    return Py_BuildValue("d", plane_norm);

fail:
    Py_XDECREF(uu);
    Py_XDECREF(vv);
    Py_XDECREF(ww);
    Py_XDECREF(amps);
    Py_XDECREF(weight);
    Py_XDECREF(plane);
    Py_XDECREF(weights_grid);
    return 0;
}


static PyObject* weighting(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_imager(capsule))) return 0;
    return Py_BuildValue("s", oskar_imager_weighting(h));
}


static PyObject* make_image(PyObject* self, PyObject* args)
{
    oskar_Imager* h;
    PyObject *obj[] = {0, 0, 0, 0, 0};
    PyArrayObject *uu = 0, *vv = 0, *ww = 0, *amps = 0, *weight = 0, *im = 0;
    int i, num_cells, num_pixels, num_vis, size = 0, status = 0, type = 0;
    int dft = 0, wproj = 0, uniform = 0, wprojplanes = -1;
    double fov_deg = 0.0, norm = 0.0;
    const char *weighting_type = 0, *algorithm_type = 0;
    oskar_Mem *uu_c, *vv_c, *ww_c, *amp_c, *weight_c, *plane;
    oskar_Mem *weights_grid = 0;
    npy_intp dims[2];

    /* Parse inputs. */
    if (!PyArg_ParseTuple(args, "OOOOdissOi",
            &obj[0], &obj[1], &obj[2], &obj[3], &fov_deg, &size,
            &weighting_type, &algorithm_type, &obj[4], &wprojplanes))
        return 0;

    /* Make sure input objects are arrays. Convert if required. */
    uu     = (PyArrayObject*) PyArray_FROM_OF(obj[0], NPY_ARRAY_IN_ARRAY);
    vv     = (PyArrayObject*) PyArray_FROM_OF(obj[1], NPY_ARRAY_IN_ARRAY);
    ww     = (PyArrayObject*) PyArray_FROM_OF(obj[2], NPY_ARRAY_IN_ARRAY);
    amps   = (PyArrayObject*) PyArray_FROM_OF(obj[3], NPY_ARRAY_IN_ARRAY);
    if (!uu || !vv || !ww || !amps)
        goto fail;

    /* Check if weights are present. */
    if (obj[4] != Py_None)
    {
        weight = (PyArrayObject*) PyArray_FROM_OF(obj[4], NPY_ARRAY_IN_ARRAY);
        if (!weight) goto fail;
    }

    /* Check dimensions. */
    num_pixels = size * size;
    if (PyArray_NDIM(uu) != 1 || PyArray_NDIM(vv) != 1 ||
            PyArray_NDIM(ww) != 1 || PyArray_NDIM(amps) != 1)
    {
        PyErr_SetString(PyExc_RuntimeError, "Input data arrays must be 1D.");
        goto fail;
    }
    num_vis = (int) PyArray_SIZE(amps);
    if (num_vis != (int) PyArray_SIZE(uu) ||
            num_vis != (int) PyArray_SIZE(vv) ||
            num_vis != (int) PyArray_SIZE(ww))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input data dimension mismatch.");
        goto fail;
    }
    if (weight && (num_vis != (int) PyArray_SIZE(weight)))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input data dimension mismatch.");
        goto fail;
    }

    /* Get precision of complex visibility data. */
    if (!PyArray_ISCOMPLEX(amps))
    {
        PyErr_SetString(PyExc_RuntimeError,
                "Input visibility data must be complex.");
        goto fail;
    }
    type = oskar_type_precision(oskar_type_from_numpy(amps));

    /* Pointers to input/output arrays. */
    uu_c = oskar_mem_create_alias_from_raw(PyArray_DATA(uu),
            oskar_type_from_numpy(uu), OSKAR_CPU, num_vis, &status);
    vv_c = oskar_mem_create_alias_from_raw(PyArray_DATA(vv),
            oskar_type_from_numpy(vv), OSKAR_CPU, num_vis, &status);
    ww_c = oskar_mem_create_alias_from_raw(PyArray_DATA(ww),
            oskar_type_from_numpy(ww), OSKAR_CPU, num_vis, &status);
    amp_c = oskar_mem_create_alias_from_raw(PyArray_DATA(amps),
            oskar_type_from_numpy(amps), OSKAR_CPU, num_vis, &status);
    if (weight)
    {
        weight_c = oskar_mem_create_alias_from_raw(PyArray_DATA(weight),
                oskar_type_from_numpy(weight), OSKAR_CPU, num_vis, &status);
    }
    else
    {
        /* Set weights to 1 if not supplied. */
        weight_c = oskar_mem_create(type, OSKAR_CPU, num_vis, &status);
        oskar_mem_set_value_real(weight_c, 1.0, 0, num_vis, &status);
    }

    /* Create and set up the imager. */
    h = oskar_imager_create(type, &status);
    oskar_imager_set_fov(h, fov_deg);
    oskar_imager_set_size(h, size, &status);
    oskar_imager_set_algorithm(h, algorithm_type, &status);
    oskar_imager_set_num_w_planes(h, wprojplanes);
    oskar_imager_set_weighting(h, weighting_type, &status);

    /* Check for DFT, W-projection or uniform weighting. */
    if (!strncmp(algorithm_type, "DFT", 3) ||
            !strncmp(algorithm_type, "dft", 3))
        dft = 1;
    if (!strncmp(algorithm_type, "W", 1) ||
            !strncmp(algorithm_type, "w", 1))
        wproj = 1;
    if (!strncmp(weighting_type, "U", 1) ||
            !strncmp(weighting_type, "u", 1))
        uniform = 1;

    /* Allow threads. */
    Py_BEGIN_ALLOW_THREADS

    /* Supply the coordinates first, if required. */
    if (wproj || uniform)
    {
        weights_grid = oskar_mem_create(type, OSKAR_CPU, num_pixels, &status);
        oskar_imager_set_coords_only(h, 1);
        oskar_imager_update_plane(h, num_vis, uu_c, vv_c, ww_c, 0, weight_c,
                0, 0, weights_grid, &status);
        oskar_imager_set_coords_only(h, 0);
    }

    /* Initialise the algorithm to get the plane size. */
    oskar_imager_check_init(h, &status);
    num_cells = oskar_imager_plane_size(h);
    num_cells *= num_cells;

    /* Make the image. */
    plane = oskar_mem_create((dft ? type : (type | OSKAR_COMPLEX)), OSKAR_CPU,
            num_cells, &status);
    oskar_imager_update_plane(h, num_vis, uu_c, vv_c, ww_c, amp_c, weight_c,
            plane, &norm, weights_grid, &status);
    oskar_imager_finalise_plane(h, plane, norm, &status);
    oskar_imager_trim_image(plane, oskar_imager_plane_size(h), size, &status);

    /* Disallow threads. */
    Py_END_ALLOW_THREADS

    /* Free temporaries. */
    oskar_mem_free(uu_c, &status);
    oskar_mem_free(vv_c, &status);
    oskar_mem_free(ww_c, &status);
    oskar_mem_free(amp_c, &status);
    oskar_mem_free(weight_c, &status);
    oskar_mem_free(weights_grid, &status);
    oskar_imager_free(h, &status);

    /* Copy the data out. */
    dims[0] = size;
    dims[1] = size;
    im = (PyArrayObject*)PyArray_SimpleNew(2, dims,
            oskar_type_is_double(type) ? NPY_DOUBLE : NPY_FLOAT);
    memcpy(PyArray_DATA(im), oskar_mem_void_const(plane),
            num_pixels * oskar_mem_element_size(type));
    oskar_mem_free(plane, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "make_image() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        goto fail;
    }

    /* Return image to the python workspace. */
    Py_XDECREF(uu);
    Py_XDECREF(vv);
    Py_XDECREF(ww);
    Py_XDECREF(amps);
    Py_XDECREF(weight);
    return Py_BuildValue("N", im); /* Don't increment refcount. */

fail:
    Py_XDECREF(uu);
    Py_XDECREF(vv);
    Py_XDECREF(ww);
    Py_XDECREF(amps);
    Py_XDECREF(weight);
    return 0;
}


/* Method table. */
static PyMethodDef methods[] =
{
        {"algorithm", (PyCFunction)algorithm, METH_VARARGS, "algorithm()"},
        {"cellsize", (PyCFunction)cellsize, METH_VARARGS, "cellsize()"},
        {"channel_end", (PyCFunction)channel_end,
                METH_VARARGS, "channel_end()"},
        {"channel_snapshots", (PyCFunction)channel_snapshots,
                METH_VARARGS, "channel_snapshots()"},
        {"channel_start", (PyCFunction)channel_start,
                METH_VARARGS, "channel_start()"},
        {"check_init", (PyCFunction)check_init, METH_VARARGS, "check_init()"},
        {"coords_only", (PyCFunction)coords_only,
                METH_VARARGS, "coords_only()"},
        {"create", (PyCFunction)create, METH_VARARGS, "create(type)"},
        {"finalise", (PyCFunction)finalise,
                METH_VARARGS, "finalise(image)"},
        {"finalise_plane", (PyCFunction)finalise_plane,
                METH_VARARGS, "finalise_plane(plane, plane_norm)"},
        {"fov", (PyCFunction)fov, METH_VARARGS, "fov()"},
        {"image_size", (PyCFunction)image_size, METH_VARARGS, "image_size()"},
        {"image_type", (PyCFunction)image_type, METH_VARARGS, "image_type()"},
        {"input_file", (PyCFunction)input_file, METH_VARARGS, "input_file()"},
        {"ms_column", (PyCFunction)ms_column, METH_VARARGS, "ms_column()"},
        {"make_image", (PyCFunction)make_image, METH_VARARGS,
                "make_image(uu, vv, ww, amp, weight, fov_deg, size)"},
        {"num_w_planes", (PyCFunction)num_w_planes,
                METH_VARARGS, "num_w_planes()"},
        {"output_root", (PyCFunction)output_root,
                METH_VARARGS, "output_root()"},
        {"plane_size", (PyCFunction)plane_size, METH_VARARGS, "plane_size()"},
        {"reset_cache", (PyCFunction)reset_cache,
                METH_VARARGS, "reset_cache()"},
        {"run", (PyCFunction)run, METH_VARARGS, "run()"},
        {"set_algorithm", (PyCFunction)set_algorithm,
                METH_VARARGS, "set_algorithm(type)"},
        {"set_cellsize", (PyCFunction)set_cellsize,
                METH_VARARGS, "set_cellsize(value)"},
        {"set_channel_end", (PyCFunction)set_channel_end,
                METH_VARARGS, "set_channel_end(value)"},
        {"set_channel_snapshots", (PyCFunction)set_channel_snapshots,
                METH_VARARGS, "set_channel_snapshots(value)"},
        {"set_channel_start", (PyCFunction)set_channel_start,
                METH_VARARGS, "set_channel_start(value)"},
        {"set_coords_only", (PyCFunction)set_coords_only,
                METH_VARARGS, "set_coords_only(flag)"},
        {"set_default_direction", (PyCFunction)set_default_direction,
                METH_VARARGS, "set_default_direction()"},
        {"set_direction", (PyCFunction)set_direction,
                METH_VARARGS, "set_direction(ra_deg, dec_deg)"},
        {"set_fft_on_gpu", (PyCFunction)set_fft_on_gpu,
                METH_VARARGS, "set_fft_on_gpu(value)"},
        {"set_fov", (PyCFunction)set_fov, METH_VARARGS, "set_fov(value)"},
        {"set_grid_kernel", (PyCFunction)set_grid_kernel,
                METH_VARARGS, "set_grid_kernel(type, support, oversample)"},
        {"set_image_size", (PyCFunction)set_image_size,
                METH_VARARGS, "set_image_size(value)"},
        {"set_image_type", (PyCFunction)set_image_type,
                METH_VARARGS, "set_image_type(type)"},
        {"set_input_file", (PyCFunction)set_input_file,
                METH_VARARGS, "set_input_file(filename)"},
        {"set_ms_column", (PyCFunction)set_ms_column,
                METH_VARARGS, "set_ms_column(column)"},
        {"set_num_w_planes", (PyCFunction)set_num_w_planes,
                METH_VARARGS, "set_num_w_planes(value)"},
        {"set_output_root", (PyCFunction)set_output_root,
                METH_VARARGS, "set_output_root(filename)"},
        {"set_size", (PyCFunction)set_size, METH_VARARGS, "set_size(value)"},
        {"set_time_end", (PyCFunction)set_time_end,
                METH_VARARGS, "set_time_end(value)"},
        {"set_time_snapshots", (PyCFunction)set_time_snapshots,
                METH_VARARGS, "set_time_snapshots(value)"},
        {"set_time_start", (PyCFunction)set_time_start,
                METH_VARARGS, "set_time_start(value)"},
        {"set_vis_frequency", (PyCFunction)set_vis_frequency, METH_VARARGS,
                "set_vis_frequency(ref_hz, inc_hz, num_channels)"},
        {"set_vis_phase_centre", (PyCFunction)set_vis_phase_centre,
                METH_VARARGS, "set_vis_phase_centre(ra_deg, dec_deg)"},
        {"set_vis_time", (PyCFunction)set_vis_time,
                METH_VARARGS, "set_vis_time(ref_mjd_utc, inc_sec, num_times)"},
        {"set_weighting", (PyCFunction)set_weighting,
                METH_VARARGS, "set_weighting(type)"},
        {"size", (PyCFunction)size, METH_VARARGS, "size()"},
        {"time_end", (PyCFunction)time_end, METH_VARARGS, "time_end()"},
        {"time_snapshots", (PyCFunction)time_snapshots,
                METH_VARARGS, "time_snapshots()"},
        {"time_start", (PyCFunction)time_start, METH_VARARGS, "time_start()"},
        {"update", (PyCFunction)update, METH_VARARGS,
                "update(num_baselines, uu, vv, ww, amps, weight, "
                "num_pols, start_time, end_time, start_chan, end_chan)"},
        {"update_from_block", (PyCFunction)update_from_block,
                METH_VARARGS, "update_from_block(vis_header, vis_block)"},
        {"update_plane", (PyCFunction)update_plane, METH_VARARGS,
                "update_plane(uu, vv, ww, amps, weight, plane, plane_norm)"},
        {"weighting", (PyCFunction)weighting, METH_VARARGS, "weighting()"},
        {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_imager_lib",      /* m_name */
        module_doc,         /* m_doc */
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
    m = Py_InitModule3("_imager_lib", methods, module_doc);
#endif
    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__imager_lib(void)
{
    import_array();
    return moduleinit();
}
#else
/* The init function name has to match that of the compiled module
 * with the pattern 'init<module name>'. This module is called '_imager_lib' */
PyMODINIT_FUNC init_imager_lib(void)
{
    import_array();
    moduleinit();
    return;
}
#endif

