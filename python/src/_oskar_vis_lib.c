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

/* http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html */
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <numpy/arrayobject.h>

#include <oskar_vis.h>

static const char* name = "oskar_Vis";

void vis_free(PyObject* ptr)
{
    int status = OSKAR_SUCCESS;
    printf("PyCapsule destructor for oskar_Vis called! (status = %i)\n", status);
    fflush(stdout);
    oskar_Vis* vis = (oskar_Vis*)PyCapsule_GetPointer(ptr, "oskar_Vis");
    if (vis) oskar_vis_free(vis, &status);
}

static inline oskar_Vis* tuple_to_vis(PyObject* objs, PyObject* args)
{
    PyObject* vis_ = NULL;
    if (!PyArg_ParseTuple(args, "O", &vis_))
        return NULL;
    if (!PyCapsule_CheckExact(vis_)) {
        printf("Input argument not a PyCapsule object!\n");
        return NULL;
    }
    oskar_Vis* vis = (oskar_Vis*)PyCapsule_GetPointer(vis_, "oskar_Vis");
    if (!vis) {
        printf("Unable to convert PyCapsule object to pointer.\n");
        return NULL;
    }
    return vis;
}

static PyObject* vis_read(PyObject* self, PyObject* args)
{
    const char* filename;
    if (!PyArg_ParseTuple(args, "s", &filename))
        return NULL;
    int status = 0;
    oskar_Binary* h = oskar_binary_create(filename, 'r', &status);
    oskar_Vis* vis = oskar_vis_read(h, &status);
    oskar_binary_free(h);
    PyObject* vis_ = PyCapsule_New((void*)vis, name, (PyCapsule_Destructor)vis_free);
    return Py_BuildValue("Ni", vis_, status);
}

static PyObject* get_num_baselines(PyObject* self, PyObject* args)
{
    oskar_Vis* vis = tuple_to_vis(self, args);
    return Py_BuildValue("i", oskar_vis_num_baselines(vis));
}

static PyObject* get_lon(PyObject* self, PyObject* args)
{
    oskar_Vis* vis = tuple_to_vis(self, args);
    return Py_BuildValue("d", oskar_vis_telescope_lon_deg(vis));
}

static PyObject* get_lat(PyObject* self, PyObject* args)
{
    oskar_Vis* vis = tuple_to_vis(self, args);
    return Py_BuildValue("d", oskar_vis_telescope_lat_deg(vis));
}

static PyObject* get_num_channels(PyObject* self, PyObject* args)
{
    oskar_Vis* vis = tuple_to_vis(self, args);
    return Py_BuildValue("i", oskar_vis_num_channels(vis));
}

static PyObject* get_num_times(PyObject* self, PyObject* args)
{
    oskar_Vis* vis = tuple_to_vis(self, args);
    return Py_BuildValue("i", oskar_vis_num_times(vis));
}

static inline PyArrayObject* mem_to_PyArrayObject(const oskar_Mem* mem)
{
    int status = OSKAR_SUCCESS;
    PyArrayObject* rtn = 0;

    size_t length = oskar_mem_length(mem);
    oskar_Mem* mem_ = oskar_mem_create_copy(mem, OSKAR_CPU, &status);
    oskar_Mem* data_ = oskar_mem_convert_precision(mem_, OSKAR_DOUBLE, &status);

    int type = oskar_mem_type(mem);

    switch (type)
    {
        case OSKAR_DOUBLE:
        case OSKAR_SINGLE:
        {
            npy_intp dims = length;
            rtn = (PyArrayObject*)PyArray_SimpleNew(1, &dims, NPY_DOUBLE);
            size_t size = length * sizeof(double);
            memcpy(PyArray_DATA(rtn), oskar_mem_void(data_), size);
            break;
        }
        case OSKAR_DOUBLE_COMPLEX:
        case OSKAR_SINGLE_COMPLEX:
        {
            break;
        }
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
        case OSKAR_SINGLE_COMPLEX_MATRIX:
        {
            npy_intp dims[2] = {length,4};
            rtn = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
            size_t size = length * 8 * sizeof(double);
            memcpy(PyArray_DATA(rtn), oskar_mem_void(data_), size);
            break;
        }
        default:
            break;
    };
    oskar_mem_free(mem_, &status);
    oskar_mem_free(data_, &status);
    return rtn;
}

static PyObject* get_station_coords(PyObject* self, PyObject* args)
{
    oskar_Vis* vis = tuple_to_vis(self, args);
    PyArrayObject* x_ = mem_to_PyArrayObject(oskar_vis_station_x_offset_ecef_metres_const(vis));
    PyArrayObject* y_ = mem_to_PyArrayObject(oskar_vis_station_y_offset_ecef_metres_const(vis));
    PyArrayObject* z_ = mem_to_PyArrayObject(oskar_vis_station_z_offset_ecef_metres_const(vis));
    return Py_BuildValue("OOO", x_, y_, z_);
}

static PyObject* get_baseline_coords(PyObject* self, PyObject* args)
{
    oskar_Vis* vis = tuple_to_vis(self, args);
    PyArrayObject* uu_ = mem_to_PyArrayObject(oskar_vis_baseline_uu_metres_const(vis));
    PyArrayObject* vv_ = mem_to_PyArrayObject(oskar_vis_baseline_vv_metres_const(vis));
    PyArrayObject* ww_ = mem_to_PyArrayObject(oskar_vis_baseline_ww_metres_const(vis));
    return Py_BuildValue("OOO", uu_, vv_, ww_);
}

static PyObject* get_amplitude(PyObject* self, PyObject* args)
{
    oskar_Vis* vis = tuple_to_vis(self, args);
    return Py_BuildValue("O", mem_to_PyArrayObject(oskar_vis_amplitude_const(vis)));
}

/* Methods table */
static PyMethodDef oskar_vis_lib_methods[] =
{
    {
            "read",
            (PyCFunction)vis_read,
            METH_VARARGS,
            "read(filename)"
    },
    {
            "num_baselines",
            (PyCFunction)get_num_baselines,
            METH_VARARGS,
            "num_baselines(vis)"
    },
    {
            "num_channels",
            (PyCFunction)get_num_channels,
            METH_VARARGS,
            "num_channels(vis)"
    },
    {
            "num_times",
            (PyCFunction)get_num_times,
            METH_VARARGS,
            "num_times(vis)"
    },
    {
            "station_coords",
            (PyCFunction)get_station_coords,
            METH_VARARGS,
            "station_coords(vis)"
    },
    {
            "lon",
            (PyCFunction)get_lon,
            METH_VARARGS,
            "lon(vis)"
    },
    {
            "lat",
            (PyCFunction)get_lat,
            METH_VARARGS,
            "lat(vis)"
    },
    {
            "baseline_coords",
            (PyCFunction)get_baseline_coords,
            METH_VARARGS,
            "baseline_coords(vis)"
    },
    {
            "amplitude",
            (PyCFunction)get_amplitude,
            METH_VARARGS,
            "amplitude(vis)"
    },
    { NULL, NULL, 0, NULL }
};

/* Initialisation function (note: has to be called called init<filename>) */
PyMODINIT_FUNC init_vis_lib(void)
{
    Py_InitModule3("_vis_lib", oskar_vis_lib_methods, "docstring...");

    /* Import the use of numpy array objects */
    import_array();
}
