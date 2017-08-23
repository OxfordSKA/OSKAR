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
        "This module provides an interface to an OSKAR visibility header.";
static const char name[] = "oskar_VisHeader";

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


static void vis_header_free(PyObject* capsule)
{
    int status = 0;
    oskar_vis_header_free((oskar_VisHeader*)
            get_handle(capsule, name), &status);
}

static PyObject* amp_type(PyObject* self, PyObject* args)
{
    oskar_VisHeader* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisHeader*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_vis_header_amp_type(h));
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


static PyObject* channel_bandwidth_hz(PyObject* self, PyObject* args)
{
    oskar_VisHeader* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisHeader*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_vis_header_channel_bandwidth_hz(h));
}


static PyObject* coord_precision(PyObject* self, PyObject* args)
{
    oskar_VisHeader* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisHeader*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_vis_header_coord_precision(h));
}


static PyObject* freq_start_hz(PyObject* self, PyObject* args)
{
    oskar_VisHeader* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisHeader*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_vis_header_freq_start_hz(h));
}


static PyObject* freq_inc_hz(PyObject* self, PyObject* args)
{
    oskar_VisHeader* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisHeader*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_vis_header_freq_inc_hz(h));
}


static PyObject* max_channels_per_block(PyObject* self, PyObject* args)
{
    oskar_VisHeader* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisHeader*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_vis_header_max_channels_per_block(h));
}


static PyObject* max_times_per_block(PyObject* self, PyObject* args)
{
    oskar_VisHeader* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisHeader*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_vis_header_max_times_per_block(h));
}


static PyObject* num_channels_total(PyObject* self, PyObject* args)
{
    oskar_VisHeader* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisHeader*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_vis_header_num_channels_total(h));
}


static PyObject* num_stations(PyObject* self, PyObject* args)
{
    oskar_VisHeader* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisHeader*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_vis_header_num_stations(h));
}


static PyObject* num_tags_per_block(PyObject* self, PyObject* args)
{
    oskar_VisHeader* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisHeader*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_vis_header_num_tags_per_block(h));
}


static PyObject* num_times_total(PyObject* self, PyObject* args)
{
    oskar_VisHeader* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisHeader*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_vis_header_num_times_total(h));
}


static PyObject* phase_centre_ra_deg(PyObject* self, PyObject* args)
{
    oskar_VisHeader* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisHeader*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_vis_header_phase_centre_ra_deg(h));
}


static PyObject* phase_centre_dec_deg(PyObject* self, PyObject* args)
{
    oskar_VisHeader* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisHeader*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_vis_header_phase_centre_dec_deg(h));
}


static PyObject* read_header(PyObject* self, PyObject* args)
{
    oskar_VisHeader* h = 0;
    oskar_Binary* b = 0;
    PyObject *binary = 0, *capsule = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &binary)) return 0;

    /* Get a handle to the binary file from the supplied capsule. */
    if (!(b = (oskar_Binary*) get_handle(binary, "oskar_Binary"))) return 0;

    /* Read the header. */
    h = oskar_vis_header_read(b, &status);

    /* Check for errors. */
    if (!h || status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_vis_header_read() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        oskar_vis_header_free(h, &status);
        return 0;
    }

    /* Store the header in a new capsule and return it. */
    capsule = PyCapsule_New((void*)h, name,
            (PyCapsule_Destructor)vis_header_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* time_start_mjd_utc(PyObject* self, PyObject* args)
{
    oskar_VisHeader* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisHeader*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_vis_header_time_start_mjd_utc(h));
}


static PyObject* time_inc_sec(PyObject* self, PyObject* args)
{
    oskar_VisHeader* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisHeader*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_vis_header_time_inc_sec(h));
}


static PyObject* time_average_sec(PyObject* self, PyObject* args)
{
    oskar_VisHeader* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisHeader*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_vis_header_time_average_sec(h));
}


/* Method table. */
static PyMethodDef methods[] =
{
        {"amp_type", (PyCFunction)amp_type, METH_VARARGS, "amp_type()"},
        {"capsule_name", (PyCFunction)capsule_name,
                METH_VARARGS, "capsule_name()"},
        {"channel_bandwidth_hz", (PyCFunction)channel_bandwidth_hz,
                METH_VARARGS, "channel_bandwidth_hz()"},
        {"coord_precision", (PyCFunction)coord_precision,
                METH_VARARGS, "coord_precision()"},
        {"freq_start_hz", (PyCFunction)freq_start_hz,
                METH_VARARGS, "freq_start_hz()"},
        {"freq_inc_hz", (PyCFunction)freq_inc_hz,
                METH_VARARGS, "freq_inc_hz()"},
        {"max_channels_per_block", (PyCFunction)max_channels_per_block,
                METH_VARARGS, "max_channels_per_block()"},
        {"max_times_per_block", (PyCFunction)max_times_per_block,
                METH_VARARGS, "max_times_per_block()"},
        {"num_channels_total", (PyCFunction)num_channels_total,
                METH_VARARGS, "num_channels_total()"},
        {"num_stations", (PyCFunction)num_stations,
                METH_VARARGS, "num_stations()"},
        {"num_tags_per_block", (PyCFunction)num_tags_per_block,
                METH_VARARGS, "num_tags_per_block()"},
        {"num_times_total", (PyCFunction)num_times_total,
                METH_VARARGS, "num_times_total()"},
        {"phase_centre_ra_deg", (PyCFunction)phase_centre_ra_deg,
                METH_VARARGS, "phase_centre_ra_deg()"},
        {"phase_centre_dec_deg", (PyCFunction)phase_centre_dec_deg,
                METH_VARARGS, "phase_centre_dec_deg()"},
        {"read_header", (PyCFunction)read_header,
                METH_VARARGS, "read_header(binary_file_handle)"},
        {"time_start_mjd_utc", (PyCFunction)time_start_mjd_utc,
                METH_VARARGS, "time_start_mjd_utc()"},
        {"time_inc_sec", (PyCFunction)time_inc_sec,
                METH_VARARGS, "time_inc_sec()"},
        {"time_average_sec", (PyCFunction)time_average_sec,
                METH_VARARGS, "time_average_sec()"},
        {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_vis_header_lib",  /* m_name */
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
    m = Py_InitModule3("_vis_header_lib", methods, module_doc);
#endif
    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__vis_header_lib(void)
{
    import_array();
    return moduleinit();
}
#else
/* The init function name has to match that of the compiled module
 * with the pattern 'init<module name>'. This module is called '_vis_header_lib' */
PyMODINIT_FUNC init_vis_header_lib(void)
{
    import_array();
    moduleinit();
    return;
}
#endif

