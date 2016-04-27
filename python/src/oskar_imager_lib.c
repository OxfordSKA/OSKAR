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

static const char* module_doc =
        "This module provides an interface to the OSKAR imager.";
static const char* name = "oskar_Imager";

static void imager_free(PyObject* capsule)
{
    int status = 0;
    oskar_Imager* h = (oskar_Imager*) PyCapsule_GetPointer(capsule, name);
    oskar_imager_free(h, &status);
}


static oskar_Imager* get_handle(PyObject* capsule)
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
                "Unable to convert PyCapsule object to pointer.");
        return 0;
    }
    return h;
}


static int oskar_type_from_numpy(PyArrayObject* arr)
{
    int type = 0;
    switch (PyArray_TYPE(arr))
    {
    case NPY_INT:     type = OSKAR_INT; break;
    case NPY_FLOAT:   type = OSKAR_SINGLE; break;
    case NPY_DOUBLE:  type = OSKAR_DOUBLE; break;
    case NPY_CFLOAT:  type = OSKAR_SINGLE_COMPLEX; break;
    case NPY_CDOUBLE: type = OSKAR_DOUBLE_COMPLEX; break;
    }
    return type;
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
    return Py_BuildValue("Ni", capsule, status); /* Don't increment refcount. */
}


static PyObject* finalise(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject *obj[] = {0, 0};
    PyArrayObject* plane = 0;
    oskar_Mem* plane_c = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "OO", &obj[0], &obj[1])) return 0;
    if (!(h = get_handle(obj[0]))) return 0;

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
    oskar_imager_finalise(h, plane_c, &status);
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
    return Py_BuildValue("i", status);

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
    if (!(h = get_handle(obj[0]))) return 0;

    /* Get the supplied plane. */
    plane = (PyArrayObject*) PyArray_FROM_OF(obj[1], NPY_ARRAY_OUT_ARRAY);
    if (!plane) goto fail;
    plane_c = oskar_mem_create_alias_from_raw(PyArray_DATA(plane),
            oskar_type_from_numpy(plane), OSKAR_CPU,
            PyArray_SIZE(plane), &status);

    /* Finalise the plane. */
    oskar_imager_finalise_plane(h, plane_c, plane_norm, &status);
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
    return Py_BuildValue("i", status);

fail:
    Py_XDECREF(plane);
    return 0;
}


static PyObject* reset_cache(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_reset_cache(h, &status);
    return Py_BuildValue("i", status);
}


static PyObject* run(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* filename = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &filename)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_run(h, filename, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_run() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("i", status);
}


static PyObject* set_algorithm(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* type = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &type)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_algorithm(h, type, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_set_algorithm() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("i", status);
}


static PyObject* set_channel_range(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int start = 0, end = 0, snapshots = 0;
    if (!PyArg_ParseTuple(args, "Oiii", &capsule, &start, &end, &snapshots))
        return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_channel_range(h, start, end, snapshots);
    return Py_BuildValue("");
}


static PyObject* set_default_direction(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_default_direction(h);
    return Py_BuildValue("");
}


static PyObject* set_direction(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    double ra = 0.0, dec = 0.0;
    if (!PyArg_ParseTuple(args, "Odd", &capsule, &ra, &dec)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_direction(h, ra, dec);
    return Py_BuildValue("");
}


static PyObject* set_fov(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    double fov = 0.0;
    if (!PyArg_ParseTuple(args, "Od", &capsule, &fov)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_fov(h, fov);
    return Py_BuildValue("");
}


static PyObject* set_fft_on_gpu(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_fft_on_gpu(h, value);
    return Py_BuildValue("");
}


static PyObject* set_image_type(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* type = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &type)) return 0;
    if (!(h = get_handle(capsule))) return 0;
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


static PyObject* set_grid_kernel(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0, support = 0, oversample = 0;
    const char* type = 0;
    if (!PyArg_ParseTuple(args, "Osii", &capsule, &type, &support, &oversample))
        return 0;
    if (!(h = get_handle(capsule))) return 0;
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


static PyObject* set_ms_column(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* column = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &column)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_ms_column(h, column, &status);
    return Py_BuildValue("i", status);
}


static PyObject* set_output_root(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* filename = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &filename)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_output_root(h, filename, &status);
    return Py_BuildValue("i", status);
}


static PyObject* set_size(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int size = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &size)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_size(h, size);
    return Py_BuildValue("");
}


static PyObject* set_time_range(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int start = 0, end = 0, snapshots = 0;
    if (!PyArg_ParseTuple(args, "Oiii", &capsule, &start, &end, &snapshots))
        return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_time_range(h, start, end, snapshots);
    return Py_BuildValue("");
}


static PyObject* set_vis_frequency(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0, num = 0;
    double ref = 0.0, inc = 0.0;
    if (!PyArg_ParseTuple(args, "Oddi", &capsule, &ref, &inc, &num)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_vis_frequency(h, ref, inc, num, &status);
    return Py_BuildValue("i", status);
}


static PyObject* set_vis_phase_centre(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    double ra = 0.0, dec = 0.0;
    if (!PyArg_ParseTuple(args, "Odd", &capsule, &ra, &dec)) return 0;
    if (!(h = get_handle(capsule))) return 0;
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
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_vis_time(h, ref, inc, num, &status);
    return Py_BuildValue("i", status);
}


static PyObject* update(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject *obj[] = {0, 0, 0, 0, 0, 0};
    oskar_Mem *uu_c, *vv_c, *ww_c, *amp_c, *weight_c;
    PyArrayObject *uu = 0, *vv = 0, *ww = 0, *amps = 0, *weight = 0;
    int start_time = 0, end_time = 0, start_chan = 0, end_chan = 0;
    int num_pols = 1, num_baselines = 0, vis_type;
    int num_times, num_chan, num_coords, num_vis, status = 0;

    /* Parse inputs. */
    if (!PyArg_ParseTuple(args, "OiOOOOOiiiii", &obj[0],
            &num_baselines, &obj[1], &obj[2], &obj[3], &obj[4], &obj[5],
            &num_pols, &start_time, &end_time, &start_chan, &end_chan))
        return 0;
    if (!(h = get_handle(obj[0]))) return 0;

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
            oskar_type_from_numpy(weight), OSKAR_CPU, num_vis, &status);

    /* Update the imager with the supplied visibility data. */
    oskar_imager_update(h, start_time, end_time, start_chan, end_chan,
            num_pols, num_baselines, uu_c, vv_c, ww_c, amp_c, weight_c,
            &status);
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
    return Py_BuildValue("i", status);

fail:
    Py_XDECREF(uu);
    Py_XDECREF(vv);
    Py_XDECREF(ww);
    Py_XDECREF(amps);
    Py_XDECREF(weight);
    return 0;
}


static PyObject* update_plane(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject *obj[] = {0, 0, 0, 0, 0, 0, 0};
    oskar_Mem *uu_c, *vv_c, *ww_c, *amp_c, *weight_c, *plane_c;
    PyArrayObject *uu = 0, *vv = 0, *ww = 0, *amps = 0, *weight = 0, *plane = 0;
    double plane_norm = 0.0;
    int num_vis, status = 0;

    /* Parse inputs: capsule, uu, vv, ww, amps, weight, plane, plane_norm. */
    if (!PyArg_ParseTuple(args, "OOOOOOOd", &obj[0],
            &obj[1], &obj[2], &obj[3], &obj[4], &obj[5], &obj[6], &plane_norm))
        return 0;
    if (!(h = get_handle(obj[0]))) return 0;

    /* Make sure input objects are arrays. Convert if required. */
    uu     = (PyArrayObject*) PyArray_FROM_OF(obj[1], NPY_ARRAY_IN_ARRAY);
    vv     = (PyArrayObject*) PyArray_FROM_OF(obj[2], NPY_ARRAY_IN_ARRAY);
    ww     = (PyArrayObject*) PyArray_FROM_OF(obj[3], NPY_ARRAY_IN_ARRAY);
    amps   = (PyArrayObject*) PyArray_FROM_OF(obj[4], NPY_ARRAY_IN_ARRAY);
    weight = (PyArrayObject*) PyArray_FROM_OF(obj[5], NPY_ARRAY_IN_ARRAY);
    plane  = (PyArrayObject*) PyArray_FROM_OF(obj[6], NPY_ARRAY_OUT_ARRAY);
    if (!uu || !vv || !ww || !amps || !weight || !plane)
        goto fail;

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

    /* Update the plane. */
    oskar_imager_update_plane(h, num_vis, uu_c, vv_c, ww_c, amp_c,
            weight_c, plane_c, &plane_norm, &status);

    /* Clean up. */
    oskar_mem_free(uu_c, &status);
    oskar_mem_free(vv_c, &status);
    oskar_mem_free(ww_c, &status);
    oskar_mem_free(amp_c, &status);
    oskar_mem_free(weight_c, &status);
    oskar_mem_free(plane_c, &status);

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
    return Py_BuildValue("d", plane_norm);

fail:
    Py_XDECREF(uu);
    Py_XDECREF(vv);
    Py_XDECREF(ww);
    Py_XDECREF(amps);
    Py_XDECREF(weight);
    Py_XDECREF(plane);
    return 0;
}


static PyObject* make_image(PyObject* self, PyObject* args)
{
    oskar_Imager* h;
    PyObject *obj[] = {0, 0, 0, 0, 0};
    PyArrayObject *uu = 0, *vv = 0, *ww = 0, *amps = 0, *weight = 0, *im = 0;
    int i, status = 0, num_vis, num_pixels, size = 0, type;
    double fov_deg = 0.0, norm = 0.0;
    oskar_Mem *uu_c, *vv_c, *ww_c, *amp_c, *weight_c, *plane;

    /* Parse inputs. */
    if (!PyArg_ParseTuple(args, "OOOOOdi",
            &obj[0], &obj[1], &obj[2], &obj[3], &obj[4], &fov_deg, &size))
        return 0;

    /* Make sure input objects are arrays. Convert if required. */
    uu     = (PyArrayObject*) PyArray_FROM_OF(obj[0], NPY_ARRAY_IN_ARRAY);
    vv     = (PyArrayObject*) PyArray_FROM_OF(obj[1], NPY_ARRAY_IN_ARRAY);
    ww     = (PyArrayObject*) PyArray_FROM_OF(obj[2], NPY_ARRAY_IN_ARRAY);
    amps   = (PyArrayObject*) PyArray_FROM_OF(obj[3], NPY_ARRAY_IN_ARRAY);
    weight = (PyArrayObject*) PyArray_FROM_OF(obj[4], NPY_ARRAY_IN_ARRAY);
    if (!uu || !vv || !ww || !amps || !weight)
        goto fail;

    /* Check dimensions. */
    if (PyArray_NDIM(uu) != 1 || PyArray_NDIM(vv) != 1 ||
            PyArray_NDIM(ww) != 1 || PyArray_NDIM(amps) != 1 ||
            PyArray_NDIM(weight) != 1)
    {
        PyErr_SetString(PyExc_RuntimeError, "Input data arrays must be 1D.");
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

    /* Get precision of complex visibility data. */
    if (!PyArray_ISCOMPLEX(amps))
    {
        PyErr_SetString(PyExc_RuntimeError,
                "Input visibility data must be complex.");
        goto fail;
    }
    type = oskar_type_precision(oskar_type_from_numpy(amps));

    /* Create the output image array. */
    num_pixels = size * size;
    npy_intp dims[] = {size, size};
    im = (PyArrayObject*)PyArray_SimpleNew(2, dims,
            oskar_type_is_double(type) ? NPY_DOUBLE : NPY_FLOAT);

    /* Create and set up the imager. */
    h = oskar_imager_create(type, &status);
    oskar_imager_set_fov(h, fov_deg);
    oskar_imager_set_size(h, size);

    /* Pointers to input/output arrays. */
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

    /* Make the image. */
    plane = oskar_mem_create(oskar_type_from_numpy(amps), OSKAR_CPU,
            num_pixels, &status);
    oskar_imager_update_plane(h, num_vis, uu_c, vv_c, ww_c, amp_c,
            weight_c, plane, &norm, &status);
    oskar_imager_finalise_plane(h, plane, norm, &status);

    /* Get the real part only. */
    if (oskar_mem_precision(plane) == OSKAR_DOUBLE)
    {
        double *t = oskar_mem_double(plane, &status);
        for (i = 0; i < num_pixels; ++i) t[i] = t[2 * i];
    }
    else
    {
        float *t = oskar_mem_float(plane, &status);
        for (i = 0; i < num_pixels; ++i) t[i] = t[2 * i];
    }
    memcpy(PyArray_DATA(im), oskar_mem_void_const(plane),
            num_pixels * oskar_mem_element_size(type));

    /* Free memory. */
    oskar_mem_free(uu_c, &status);
    oskar_mem_free(vv_c, &status);
    oskar_mem_free(ww_c, &status);
    oskar_mem_free(amp_c, &status);
    oskar_mem_free(weight_c, &status);
    oskar_imager_free(h, &status);

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
    return Py_BuildValue("O", im);

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
        {"create", (PyCFunction)create, METH_VARARGS, "create(type)"},
        {"finalise", (PyCFunction)finalise, METH_VARARGS,
                "finalise(image)"},
        {"finalise_plane", (PyCFunction)finalise_plane, METH_VARARGS,
                "finalise_plane(plane, plane_norm)"},
        {"make_image", (PyCFunction)make_image, METH_VARARGS,
                "make_image(uu, vv, ww, amp, weight, fov_deg, size)"},
        {"reset_cache", (PyCFunction)reset_cache, METH_VARARGS,
                "reset_cache()"},
        {"run", (PyCFunction)run, METH_VARARGS, "run(filename)"},
        {"set_algorithm", (PyCFunction)set_algorithm, METH_VARARGS,
                "set_algorithm(type)"},
        {"set_default_direction", (PyCFunction)set_default_direction, METH_VARARGS,
                "set_default_direction()"},
        {"set_direction", (PyCFunction)set_direction, METH_VARARGS,
                "set_direction(ra_deg, dec_deg)"},
        {"set_channel_range", (PyCFunction)set_channel_range, METH_VARARGS,
                "set_channel_range(start, end, snapshots)"},
        {"set_grid_kernel", (PyCFunction)set_grid_kernel, METH_VARARGS,
                "set_grid_kernel(type, support, oversample)"},
        {"set_image_type", (PyCFunction)set_image_type, METH_VARARGS,
                "set_image_type(type)"},
        {"set_fft_on_gpu", (PyCFunction)set_fft_on_gpu, METH_VARARGS,
                "set_fft_on_gpu(value)"},
        {"set_fov", (PyCFunction)set_fov, METH_VARARGS, "set_fov(value)"},
        {"set_ms_column", (PyCFunction)set_ms_column, METH_VARARGS,
                "set_ms_column(column)"},
        {"set_output_root", (PyCFunction)set_output_root, METH_VARARGS,
                "set_output_root(filename)"},
        {"set_size", (PyCFunction)set_size, METH_VARARGS, "set_size(value)"},
        {"set_time_range", (PyCFunction)set_time_range, METH_VARARGS,
                "set_time_range(start, end, snapshots)"},
        {"set_vis_frequency", (PyCFunction)set_vis_frequency, METH_VARARGS,
                "set_vis_frequency(ref_hz, inc_hz, num_channels)"},
        {"set_vis_phase_centre", (PyCFunction)set_vis_phase_centre, METH_VARARGS,
                "set_vis_phase_centre(ra_deg, dec_deg)"},
        {"set_vis_time", (PyCFunction)set_vis_time, METH_VARARGS,
                "set_vis_time(ref_mjd_utc, inc_sec, num_times)"},
        {"update", (PyCFunction)update, METH_VARARGS,
                "update(num_baselines, uu, vv, ww, amps, weight, "
                "num_pols, start_time, end_time, start_chan, end_chan)"},
        {"update_plane", (PyCFunction)update_plane, METH_VARARGS,
                "update_plane(uu, vv, ww, amps, weight, plane, plane_norm)"},
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

