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

#include <oskar_sim_interferometer.h>
#include <oskar_get_error_string.h>
#include <string.h>

/* http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static const char* module_doc =
        "This module provides an interface to the OSKAR simulator.";
static const char* name = "oskar_Simulator";

static void simulator_free(PyObject* capsule)
{
    int status = 0;
    oskar_Simulator* h = (oskar_Simulator*) PyCapsule_GetPointer(capsule, name);
    oskar_simulator_free(h, &status);
}


static oskar_Simulator* get_handle_simulator(PyObject* capsule)
{
    oskar_Simulator* h = 0;
    if (!PyCapsule_CheckExact(capsule))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input is not a PyCapsule object!");
        return 0;
    }
    h = (oskar_Simulator*) PyCapsule_GetPointer(capsule, name);
    if (!h)
    {
        PyErr_SetString(PyExc_RuntimeError,
                "Unable to convert PyCapsule object to oskar_Simulator.");
        return 0;
    }
    return h;
}


static oskar_Sky* get_handle_sky(PyObject* capsule)
{
    oskar_Sky* h = 0;
    if (!PyCapsule_CheckExact(capsule))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input is not a PyCapsule object!");
        return 0;
    }
    h = (oskar_Sky*) PyCapsule_GetPointer(capsule, "oskar_Sky");
    if (!h)
    {
        PyErr_SetString(PyExc_RuntimeError,
                "Unable to convert PyCapsule object to oskar_Sky.");
        return 0;
    }
    return h;
}


static oskar_Telescope* get_handle_telescope(PyObject* capsule)
{
    oskar_Telescope* h = 0;
    if (!PyCapsule_CheckExact(capsule))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input is not a PyCapsule object!");
        return 0;
    }
    h = (oskar_Telescope*) PyCapsule_GetPointer(capsule, "oskar_Telescope");
    if (!h)
    {
        PyErr_SetString(PyExc_RuntimeError,
                "Unable to convert PyCapsule object to oskar_Telescope.");
        return 0;
    }
    return h;
}


static PyObject* check_init(PyObject* self, PyObject* args)
{
    oskar_Simulator* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_simulator(capsule))) return 0;
    oskar_simulator_check_init(h, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_simulator_check_init() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* create(PyObject* self, PyObject* args)
{
    oskar_Simulator* h = 0;
    PyObject* capsule = 0;
    int status = 0, prec = 0;
    const char* type;
    if (!PyArg_ParseTuple(args, "s", &type)) return 0;
    prec = (type[0] == 'S' || type[0] == 's') ? OSKAR_SINGLE : OSKAR_DOUBLE;
    h = oskar_simulator_create(prec, &status);
    capsule = PyCapsule_New((void*)h, name,
            (PyCapsule_Destructor)simulator_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* set_observation_frequency(PyObject* self, PyObject* args)
{
    oskar_Simulator* h = 0;
    PyObject* capsule = 0;
    int num_channels;
    double start_frequency_hz, inc_hz;
    if (!PyArg_ParseTuple(args, "Oddi", &capsule,
            &start_frequency_hz, &inc_hz, &num_channels)) return 0;
    if (!(h = get_handle_simulator(capsule))) return 0;
    oskar_simulator_set_observation_frequency(h, start_frequency_hz,
            inc_hz, num_channels);
    return Py_BuildValue("");
}


static PyObject* set_observation_time(PyObject* self, PyObject* args)
{
    oskar_Simulator* h = 0;
    PyObject* capsule = 0;
    int num_times;
    double start_time_mjd_utc, length_sec, inc_sec;
    if (!PyArg_ParseTuple(args, "Oddi", &capsule,
            &start_time_mjd_utc, &length_sec, &num_times)) return 0;
    if (!(h = get_handle_simulator(capsule))) return 0;
    inc_sec = length_sec / num_times;
    oskar_simulator_set_observation_time(h, start_time_mjd_utc,
            inc_sec, num_times);
    return Py_BuildValue("");
}


static PyObject* set_output_measurement_set(PyObject* self, PyObject* args)
{
    oskar_Simulator* h = 0;
    PyObject* capsule = 0;
    const char* filename;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &filename)) return 0;
    if (!(h = get_handle_simulator(capsule))) return 0;
    oskar_simulator_set_output_measurement_set(h, filename);
    return Py_BuildValue("");
}


static PyObject* set_output_vis_file(PyObject* self, PyObject* args)
{
    oskar_Simulator* h = 0;
    PyObject* capsule = 0;
    const char* filename;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &filename)) return 0;
    if (!(h = get_handle_simulator(capsule))) return 0;
    oskar_simulator_set_output_vis_file(h, filename);
    return Py_BuildValue("");
}


static PyObject* set_sky_model(PyObject* self, PyObject* args)
{
    oskar_Simulator* h = 0;
    oskar_Sky* s = 0;
    PyObject *capsule = 0, *sm = 0;
    int status = 0, max_sources_per_chunk = 0;
    if (!PyArg_ParseTuple(args, "OOi", &capsule, &sm, &max_sources_per_chunk))
        return 0;
    if (!(h = get_handle_simulator(capsule))) return 0;
    if (!(s = get_handle_sky(sm))) return 0;
    oskar_simulator_set_sky_model(h, s, max_sources_per_chunk, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_simulator_set_sky_model() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* set_telescope_model(PyObject* self, PyObject* args)
{
    oskar_Simulator* h = 0;
    oskar_Telescope* t = 0;
    PyObject *capsule = 0, *tm = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &tm)) return 0;
    if (!(h = get_handle_simulator(capsule))) return 0;
    if (!(t = get_handle_telescope(tm))) return 0;
    oskar_simulator_set_telescope_model(h, t, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_simulator_set_telescope_model() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* reset_cache(PyObject* self, PyObject* args)
{
    oskar_Simulator* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_simulator(capsule))) return 0;
    oskar_simulator_reset_cache(h, &status);
    return Py_BuildValue("");
}


static PyObject* run(PyObject* self, PyObject* args)
{
    oskar_Simulator* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle_simulator(capsule))) return 0;
    oskar_simulator_run(h, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_simulator_run() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


/* Method table. */
static PyMethodDef methods[] =
{
        {"check_init", (PyCFunction)check_init, METH_VARARGS, "check_init()"},
        {"create", (PyCFunction)create, METH_VARARGS, "create(type)"},
        {"run", (PyCFunction)run, METH_VARARGS, "run()"},
        {"reset_cache", (PyCFunction)reset_cache, METH_VARARGS,
                "reset_cache()"},
        {"set_observation_frequency", (PyCFunction)set_observation_frequency,
                METH_VARARGS,
                "set_observation_frequency(start_freq_hz, inc_hz, "
                "num_channels)"},
        {"set_output_measurement_set",
                (PyCFunction)set_output_measurement_set, METH_VARARGS,
                "set_output_measurement_set(filename)"},
        {"set_output_vis_file",
                (PyCFunction)set_output_vis_file, METH_VARARGS,
                "set_output_vis_file(filename)"},
        {"set_sky_model", (PyCFunction)set_sky_model, METH_VARARGS,
                "set_sky_model(sky, max_sources_per_chunk)"},
        {"set_telescope_model", (PyCFunction)set_telescope_model, METH_VARARGS,
                "set_telescope_model(telescope)"},
        {"set_observation_time", (PyCFunction)set_observation_time,
                METH_VARARGS,
                "set_observation_time(start_time_mjd_utc, length_sec, "
                "num_time_steps)"},
        {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_simulator_lib",   /* m_name */
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
    m = Py_InitModule3("_simulator_lib", methods, module_doc);
#endif
    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__simulator_lib(void)
{
    import_array();
    return moduleinit();
}
#else
/* The init function name has to match that of the compiled module
 * with the pattern 'init<module name>'. This module is called '_simulator_lib' */
PyMODINIT_FUNC init_simulator_lib(void)
{
    import_array();
    moduleinit();
    return;
}
#endif

