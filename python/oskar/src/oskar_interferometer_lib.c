/*
 * Copyright (c) 2016-2017, The University of Oxford
 * All rights reserved.
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

#include <oskar.h>
#include <string.h>

/* http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static const char module_doc[] =
        "This module provides an interface to the OSKAR interferometer "
        "simulator.";
static const char name[] = "oskar_Interferometer";

static void* get_handle(PyObject* capsule, const char* name)
{
    void* h = 0;
    if (!PyCapsule_CheckExact(capsule))
    {
        PyErr_SetString(PyExc_RuntimeError, "Object is not a PyCapsule.");
        return 0;
    }
    if (!(h = PyCapsule_GetPointer(capsule, name)))
    {
        PyErr_Format(PyExc_RuntimeError, "Capsule is not of type %s.", name);
        return 0;
    }
    return h;
}


static void interferometer_free(PyObject* capsule)
{
    int status = 0;
    oskar_interferometer_free((oskar_Interferometer*)
            get_handle(capsule, name), &status);
}


static PyObject* capsule_name(PyObject* self, PyObject* args)
{
    PyObject *capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!PyCapsule_CheckExact(capsule))
    {
        PyErr_SetString(PyExc_RuntimeError, "Object is not a PyCapsule.");
        return 0;
    }
    return Py_BuildValue("s", PyCapsule_GetName(capsule));
}


static PyObject* check_init(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    oskar_interferometer_check_init(h, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_interferometer_check_init() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* coords_only(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    int flag;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    flag = oskar_interferometer_coords_only(h);
    return Py_BuildValue("O", flag ? Py_True : Py_False);
}


static PyObject* create(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    int status = 0, prec = 0;
    const char* type;
    if (!PyArg_ParseTuple(args, "s", &type)) return 0;
    prec = (type[0] == 'S' || type[0] == 's') ? OSKAR_SINGLE : OSKAR_DOUBLE;
    h = oskar_interferometer_create(prec, &status);
    capsule = PyCapsule_New((void*)h, name,
            (PyCapsule_Destructor)interferometer_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* finalise_block(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    oskar_VisBlock* b = 0;
    PyObject *capsule = 0, *block = 0;
    int block_index = 0, status = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &block_index)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;

    Py_BEGIN_ALLOW_THREADS
    b = oskar_interferometer_finalise_block(h, block_index, &status);
    Py_END_ALLOW_THREADS
    block = PyCapsule_New((void*)b, "oskar_VisBlock", NULL);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_interferometer_finalise_block() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("N", block); /* Don't increment refcount. */
}


static PyObject* finalise(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    oskar_interferometer_finalise(h, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_interferometer_finalise() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* num_devices(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_interferometer_num_devices(h));
}


static PyObject* num_gpus(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_interferometer_num_gpus(h));
}


static PyObject* num_vis_blocks(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_interferometer_num_vis_blocks(h));
}


static PyObject* reset_cache(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    oskar_interferometer_reset_cache(h, &status);
    return Py_BuildValue("");
}


static PyObject* reset_work_unit_index(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    oskar_interferometer_reset_work_unit_index(h);
    return Py_BuildValue("");
}


static PyObject* run_block(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    int block_index = 0, device_id = 0, status = 0;
    if (!PyArg_ParseTuple(args, "Oii", &capsule, &block_index, &device_id))
        return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;

    Py_BEGIN_ALLOW_THREADS
    oskar_interferometer_run_block(h, block_index, device_id, &status);
    Py_END_ALLOW_THREADS

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_interferometer_run_block() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* run(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    oskar_interferometer_run(h, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_interferometer_run() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* set_coords_only(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    int status = 0, value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    oskar_interferometer_set_coords_only(h, value, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_interferometer_set_coords_only() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* set_correlation_type(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* type;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &type)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    oskar_interferometer_set_correlation_type(h, type, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_interferometer_set_correlation_type() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* set_force_polarised_ms(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    oskar_interferometer_set_force_polarised_ms(h, value);
    return Py_BuildValue("");
}


static PyObject* set_gpus(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject *capsule = 0, *array = 0;
    PyArrayObject *gpus = 0;
    int flags, num_gpus, status = 0;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &array)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;

    /* Check if array is None. */
    if (array == Py_None)
    {
        /* Don't use GPUs. */
        oskar_interferometer_set_gpus(h, 0, 0, &status);
    }
    else
    {
        /* Get the list of GPUs. */
        flags = NPY_ARRAY_FORCECAST | NPY_ARRAY_IN_ARRAY;
        gpus = (PyArrayObject*) PyArray_FROM_OTF(array, NPY_INT, flags);
        if (!gpus) goto fail;

        /* Set the GPUs to use. */
        num_gpus = (int) PyArray_SIZE(gpus);
        if (num_gpus > 0 && ((int*) PyArray_DATA(gpus))[0] < 0)
            num_gpus = -1;
        oskar_interferometer_set_gpus(h, num_gpus, (int*) PyArray_DATA(gpus),
                &status);
    }

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_interferometer_set_gpus() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }

    Py_XDECREF(gpus);
    return Py_BuildValue("");

fail:
    Py_XDECREF(gpus);
    return 0;
}


static PyObject* set_horizon_clip(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    oskar_interferometer_set_horizon_clip(h, value);
    return Py_BuildValue("");
}


static PyObject* set_max_sources_per_chunk(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    oskar_interferometer_set_max_sources_per_chunk(h, value);
    return Py_BuildValue("");
}


static PyObject* set_max_times_per_block(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    oskar_interferometer_set_max_times_per_block(h, value);
    return Py_BuildValue("");
}


static PyObject* set_num_devices(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    oskar_interferometer_set_num_devices(h, value);
    return Py_BuildValue("");
}


static PyObject* set_observation_frequency(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    int num_channels;
    double start_frequency_hz, inc_hz;
    if (!PyArg_ParseTuple(args, "Oddi", &capsule,
            &start_frequency_hz, &inc_hz, &num_channels)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    oskar_interferometer_set_observation_frequency(h, start_frequency_hz,
            inc_hz, num_channels);
    return Py_BuildValue("");
}


static PyObject* set_observation_time(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    int num_times;
    double start_time_mjd_utc, length_sec, inc_sec;
    if (!PyArg_ParseTuple(args, "Oddi", &capsule,
            &start_time_mjd_utc, &length_sec, &num_times)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    inc_sec = length_sec / num_times;
    oskar_interferometer_set_observation_time(h, start_time_mjd_utc,
            inc_sec, num_times);
    return Py_BuildValue("");
}


static PyObject* set_output_measurement_set(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    const char* filename;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &filename)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    oskar_interferometer_set_output_measurement_set(h, filename);
    return Py_BuildValue("");
}


static PyObject* set_output_vis_file(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    const char* filename;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &filename)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    oskar_interferometer_set_output_vis_file(h, filename);
    return Py_BuildValue("");
}


static PyObject* set_settings_path(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    const char* filename;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &filename)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    oskar_interferometer_set_settings_path(h, filename);
    return Py_BuildValue("");
}


static PyObject* set_sky_model(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    oskar_Sky* s = 0;
    PyObject *capsule = 0, *sm = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &sm)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    if (!(s = (oskar_Sky*) get_handle(sm, "oskar_Sky"))) return 0;
    oskar_interferometer_set_sky_model(h, s, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_interferometer_set_sky_model() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* set_telescope_model(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    oskar_Telescope* t = 0;
    PyObject *capsule = 0, *tm = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &tm)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    if (!(t = (oskar_Telescope*) get_handle(tm, "oskar_Telescope"))) return 0;
    oskar_interferometer_set_telescope_model(h, t, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_interferometer_set_telescope_model() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* set_zero_failed_gaussians(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    oskar_interferometer_set_zero_failed_gaussians(h, value);
    return Py_BuildValue("");
}


static PyObject* vis_header(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    const oskar_VisHeader* hdr = 0;
    PyObject *capsule = 0, *header = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    hdr = oskar_interferometer_vis_header(h);

    /* Check for NULL pointer. */
    if (!hdr)
    {
        PyErr_Format(PyExc_RuntimeError,
                "Visibility header doesn't exist. Call check_init() first.");
        return 0;
    }

    header = PyCapsule_New((void*)hdr, "oskar_VisHeader", NULL);
    return Py_BuildValue("N", header); /* Don't increment refcount. */
}


static PyObject* write_block(PyObject* self, PyObject* args)
{
    oskar_Interferometer* h = 0;
    oskar_VisBlock* b = 0;
    PyObject *capsule = 0, *block = 0;
    int block_index = 0, status = 0;
    if (!PyArg_ParseTuple(args, "OOi", &capsule, &block, &block_index))
        return 0;
    if (!(h = (oskar_Interferometer*) get_handle(capsule, name))) return 0;
    if (!(b = (oskar_VisBlock*) get_handle(block, "oskar_VisBlock"))) return 0;

    Py_BEGIN_ALLOW_THREADS
    oskar_interferometer_write_block(h, b, block_index, &status);
    Py_END_ALLOW_THREADS

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_interferometer_write_block() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


/* Method table. */
static PyMethodDef methods[] =
{
        {"capsule_name", (PyCFunction)capsule_name,
                METH_VARARGS, "capsule_name()"},
        {"check_init", (PyCFunction)check_init, METH_VARARGS, "check_init()"},
        {"coords_only", (PyCFunction)coords_only,
                METH_VARARGS, "coords_only()"},
        {"create", (PyCFunction)create, METH_VARARGS, "create(type)"},
        {"finalise_block", (PyCFunction)finalise_block,
                METH_VARARGS, "finalise_block(block_index)"},
        {"finalise", (PyCFunction)finalise, METH_VARARGS, "finalise()"},
        {"num_devices", (PyCFunction)num_devices,
                METH_VARARGS, "num_devices()"},
        {"num_gpus", (PyCFunction)num_gpus, METH_VARARGS, "num_gpus()"},
        {"num_vis_blocks", (PyCFunction)num_vis_blocks,
                METH_VARARGS, "num_vis_blocks()"},
        {"reset_cache", (PyCFunction)reset_cache,
                METH_VARARGS, "reset_cache()"},
        {"reset_work_unit_index", (PyCFunction)reset_work_unit_index,
                METH_VARARGS, "reset_work_unit_index()"},
        {"run_block", (PyCFunction)run_block,
                METH_VARARGS, "run_block(block_index, gpu_id)"},
        {"run", (PyCFunction)run, METH_VARARGS, "run()"},
        {"set_coords_only", (PyCFunction)set_coords_only,
                METH_VARARGS, "set_coords_only(value)"},
        {"set_correlation_type", (PyCFunction)set_correlation_type,
                METH_VARARGS, "set_correlation_type(type)"},
        {"set_force_polarised_ms", (PyCFunction)set_force_polarised_ms,
                METH_VARARGS, "set_force_polarised_ms(value)"},
        {"set_gpus", (PyCFunction)set_gpus,
                METH_VARARGS, "set_gpus(device_ids)"},
        {"set_horizon_clip", (PyCFunction)set_horizon_clip,
                METH_VARARGS, "set_horizon_clip(value)"},
        {"set_max_sources_per_chunk", (PyCFunction)set_max_sources_per_chunk,
                METH_VARARGS, "set_max_sources_per_chunk(value)"},
        {"set_max_times_per_block", (PyCFunction)set_max_times_per_block,
                METH_VARARGS, "set_max_times_per_block(value)"},
        {"set_num_devices", (PyCFunction)set_num_devices,
                METH_VARARGS, "set_num_devices(value)"},
        {"set_observation_frequency", (PyCFunction)set_observation_frequency,
                METH_VARARGS,
                "set_observation_frequency(start_freq_hz, inc_hz, "
                "num_channels)"},
        {"set_observation_time", (PyCFunction)set_observation_time,
                METH_VARARGS,
                "set_observation_time(start_time_mjd_utc, length_sec, "
                "num_time_steps)"},
        {"set_output_measurement_set", (PyCFunction)set_output_measurement_set,
                METH_VARARGS, "set_output_measurement_set(filename)"},
        {"set_output_vis_file", (PyCFunction)set_output_vis_file,
                METH_VARARGS, "set_output_vis_file(filename)"},
        {"set_settings_path", (PyCFunction)set_settings_path,
                METH_VARARGS, "set_settings_path(filename)"},
        {"set_sky_model", (PyCFunction)set_sky_model,
                METH_VARARGS, "set_sky_model(sky)"},
        {"set_telescope_model", (PyCFunction)set_telescope_model,
                METH_VARARGS, "set_telescope_model(telescope)"},
        {"set_zero_failed_gaussians", (PyCFunction)set_zero_failed_gaussians,
                METH_VARARGS, "set_zero_failed_gaussians(value)"},
        {"vis_header", (PyCFunction)vis_header, METH_VARARGS, "vis_header()"},
        {"write_block", (PyCFunction)write_block,
                METH_VARARGS, "write_block(block_index)"},
        {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_interferometer_lib", /* m_name */
        module_doc,            /* m_doc */
        -1,                    /* m_size */
        methods                /* m_methods */
};
#endif


static PyObject* moduleinit(void)
{
    PyObject* m;
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3("_interferometer_lib", methods, module_doc);
#endif
    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__interferometer_lib(void)
{
    import_array();
    return moduleinit();
}
#else
/* The init function name has to match that of the compiled module
 * with the pattern 'init<module name>'. This module is called '_interferometer_lib' */
PyMODINIT_FUNC init_interferometer_lib(void)
{
    import_array();
    moduleinit();
    return;
}
#endif

