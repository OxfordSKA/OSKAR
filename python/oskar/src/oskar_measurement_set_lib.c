/*
 * Copyright (c) 2016, The University of Oxford
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

#include <ms/oskar_measurement_set.h>
#include <string.h>

/* http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static const char* module_doc =
        "This module provides an interface to the OSKAR Measurement Set wrapper.";
static const char* name = "oskar_MeasurementSet";

static void ms_free(PyObject* capsule)
{
    oskar_MeasurementSet* h =
            (oskar_MeasurementSet*) PyCapsule_GetPointer(capsule, name);
    oskar_ms_close(h);
}


static oskar_MeasurementSet* get_handle(PyObject* capsule)
{
    oskar_MeasurementSet* h = 0;
    if (!PyCapsule_CheckExact(capsule))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input is not a PyCapsule object!");
        return 0;
    }
    h = (oskar_MeasurementSet*) PyCapsule_GetPointer(capsule, name);
    if (!h)
    {
        PyErr_SetString(PyExc_RuntimeError,
                "Unable to convert PyCapsule object to oskar_MeasurementSet.");
        return 0;
    }
    return h;
}


static PyObject* create(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject *capsule = 0;
    int num_pols = 0, num_channels = 0, num_stations = 0;
    int write_autocorr = 0, write_crosscor = 0;
    double ref_freq_hz = 0.0, chan_width_hz = 0.0;
    const char* file_name = 0;
    if (!PyArg_ParseTuple(args, "siiiddii", &file_name, &num_stations,
            &num_channels, &num_pols, &ref_freq_hz, &chan_width_hz,
            &write_autocorr, &write_crosscor)) return 0;
    h = oskar_ms_create(file_name, "Python script", num_stations, num_channels,
            num_pols, ref_freq_hz, chan_width_hz, write_autocorr,
            write_crosscor);
    capsule = PyCapsule_New((void*)h, name, (PyCapsule_Destructor)ms_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* num_channels(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    return Py_BuildValue("i", oskar_ms_num_channels(h));
}


static PyObject* num_pols(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    return Py_BuildValue("i", oskar_ms_num_pols(h));
}


static PyObject* num_rows(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    return Py_BuildValue("i", oskar_ms_num_rows(h));
}


static PyObject* num_stations(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    return Py_BuildValue("i", oskar_ms_num_stations(h));
}


static PyObject* open(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject *capsule = 0;
    const char* file_name = 0;
    if (!PyArg_ParseTuple(args, "s", &file_name)) return 0;
    h = oskar_ms_open(file_name);
    capsule = PyCapsule_New((void*)h, name, (PyCapsule_Destructor)ms_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* read_coords(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject *capsule = 0, *tuple = 0;
    PyArrayObject *uu = 0, *vv = 0, *ww = 0;
    npy_intp dims[1];
    int start_row = 0, num_rows = 0, status = 0;
    if (!PyArg_ParseTuple(args, "Oii", &capsule, &start_row, &num_rows))
        return 0;
    if (!(h = get_handle(capsule))) return 0;

    /* Create numpy arrays to return. */
    dims[0] = num_rows;
    uu = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    vv = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    ww = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    /* Allow threads. */
    Py_BEGIN_ALLOW_THREADS

    /* Read the coordinates. */
    oskar_ms_read_coords_d(h, start_row, num_rows,
            (double*)PyArray_DATA(uu),
            (double*)PyArray_DATA(vv),
            (double*)PyArray_DATA(ww), &status);

    /* Disallow threads. */
    Py_END_ALLOW_THREADS

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_ms_read_coords() failed with code %d.", status);
        Py_XDECREF(uu);
        Py_XDECREF(vv);
        Py_XDECREF(ww);
        return 0;
    }

    /* Return a tuple of the coordinates. */
    tuple = PyTuple_Pack(3, uu, vv, ww);
    return Py_BuildValue("N", tuple); /* Don't increment refcount. */
}


static PyObject* read_vis(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject *capsule = 0;
    PyArrayObject *vis = 0;
    npy_intp dims[3];
    int start_row = 0, num_baselines = 0;
    int start_channel = 0, num_channels = 0, status = 0;
    const char* column_name = 0;
    if (!PyArg_ParseTuple(args, "Oiiiis", &capsule, &start_row, &start_channel,
            &num_channels, &num_baselines, &column_name))
        return 0;
    if (!(h = get_handle(capsule))) return 0;

    /* Create numpy array to return. */
    dims[0] = num_channels;
    dims[1] = num_baselines;
    dims[2] = oskar_ms_num_pols(h);
    vis = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_CFLOAT);

    /* Allow threads. */
    Py_BEGIN_ALLOW_THREADS

    /* Read the visibility data. */
    oskar_ms_read_vis_f(h, start_row, start_channel,
            num_channels, num_baselines, column_name,
            (float*)PyArray_DATA(vis), &status);

    /* Disallow threads. */
    Py_END_ALLOW_THREADS

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_ms_read_vis() failed with code %d.", status);
        Py_XDECREF(vis);
        return 0;
    }

    /* Return the data. */
    return Py_BuildValue("N", vis); /* Don't increment refcount. */
}


static PyObject* set_phase_centre(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject *capsule = 0;
    double lon_rad = 0.0, lat_rad = 0.0;
    if (!PyArg_ParseTuple(args, "Odd", &capsule, &lon_rad, &lat_rad))
        return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_ms_set_phase_centre(h, 0, lon_rad, lat_rad);
    return Py_BuildValue("");
}


static PyObject* write_coords(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject *capsule = 0;
    PyObject *obj[] = {0, 0, 0};
    PyArrayObject *uu = 0, *vv = 0, *ww = 0;
    int start_row = 0, num_baselines = 0;
    double exposure_sec = 0.0, interval_sec = 0.0, time_stamp = 0.0;
    if (!PyArg_ParseTuple(args, "OiiOOOddd", &capsule,
            &start_row, &num_baselines, &obj[0], &obj[1], &obj[2],
            &exposure_sec, &interval_sec, &time_stamp))
        return 0;
    if (!(h = get_handle(capsule))) return 0;

    /* Make sure input objects are arrays. Convert if required. */
    uu = (PyArrayObject*) PyArray_FROM_OF(obj[0], NPY_ARRAY_IN_ARRAY);
    vv = (PyArrayObject*) PyArray_FROM_OF(obj[1], NPY_ARRAY_IN_ARRAY);
    ww = (PyArrayObject*) PyArray_FROM_OF(obj[2], NPY_ARRAY_IN_ARRAY);
    if (!uu || !vv || !ww)
        goto fail;

    /* Check dimensions. */
    if (num_baselines != (int) PyArray_SIZE(uu) ||
            num_baselines != (int) PyArray_SIZE(vv) ||
            num_baselines != (int) PyArray_SIZE(ww))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input data dimension mismatch.");
        goto fail;
    }

    /* Allow threads. */
    Py_BEGIN_ALLOW_THREADS

    /* Write the coordinates. */
    if (PyArray_TYPE(uu) == NPY_DOUBLE)
        oskar_ms_write_coords_d(h, start_row, num_baselines,
                (const double*)PyArray_DATA(uu),
                (const double*)PyArray_DATA(vv),
                (const double*)PyArray_DATA(ww),
                exposure_sec, interval_sec, time_stamp);
    else
        oskar_ms_write_coords_f(h, start_row, num_baselines,
                (const float*)PyArray_DATA(uu),
                (const float*)PyArray_DATA(vv),
                (const float*)PyArray_DATA(ww),
                exposure_sec, interval_sec, time_stamp);

    /* Disallow threads. */
    Py_END_ALLOW_THREADS

    Py_XDECREF(uu);
    Py_XDECREF(vv);
    Py_XDECREF(ww);
    return Py_BuildValue("");

fail:
    Py_XDECREF(uu);
    Py_XDECREF(vv);
    Py_XDECREF(ww);
    return 0;
}


static PyObject* write_vis(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject *capsule = 0;
    PyObject *obj = 0;
    PyArrayObject *vis = 0;
    int start_row = 0, start_channel = 0;
    int num_channels = 0, num_baselines = 0, num_pols = 0;
    if (!PyArg_ParseTuple(args, "OiiiiO", &capsule, &start_row,
            &start_channel, &num_channels, &num_baselines, &obj))
        return 0;
    if (!(h = get_handle(capsule))) return 0;

    /* Make sure input objects are arrays. Convert if required. */
    vis = (PyArrayObject*) PyArray_FROM_OF(obj, NPY_ARRAY_IN_ARRAY);
    if (!vis)
        goto fail;

    /* Get precision of complex visibility data. */
    if (!PyArray_ISCOMPLEX(vis))
    {
        PyErr_SetString(PyExc_RuntimeError,
                "Input visibility data must be complex.");
        goto fail;
    }

    /* Check dimensions. */
    num_pols = oskar_ms_num_pols(h);
    if (num_baselines * num_channels * num_pols != (int) PyArray_SIZE(vis))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input data dimension mismatch.");
        goto fail;
    }

    /* Allow threads. */
    Py_BEGIN_ALLOW_THREADS

    /* Write the visibilities. */
    if (PyArray_TYPE(vis) == NPY_DOUBLE)
        oskar_ms_write_vis_d(h, start_row, start_channel, num_channels,
                num_baselines, (const double*)PyArray_DATA(vis));
    else
        oskar_ms_write_vis_f(h, start_row, start_channel, num_channels,
                num_baselines, (const float*)PyArray_DATA(vis));

    /* Disallow threads. */
    Py_END_ALLOW_THREADS

    Py_XDECREF(vis);
    return Py_BuildValue("");

fail:
    Py_XDECREF(vis);
    return 0;
}


/* Method table. */
static PyMethodDef methods[] =
{
        {"create", (PyCFunction)create,
                METH_VARARGS, "create(file_name, num_stations, num_channels, "
                "num_pols, ref_freq_hz, chan_width_hz, "
                "write_autocorr, write_crosscor)"},
        {"num_channels", (PyCFunction)num_channels,
                METH_VARARGS, "num_channels()"},
        {"num_pols", (PyCFunction)num_pols, METH_VARARGS, "num_pols()"},
        {"num_rows", (PyCFunction)num_rows, METH_VARARGS, "num_rows()"},
        {"num_stations", (PyCFunction)num_stations,
                METH_VARARGS, "num_stations()"},
        {"open", (PyCFunction)open, METH_VARARGS, "open(filename)"},
        {"read_coords", (PyCFunction)read_coords,
                METH_VARARGS, "read_coords(start_row, num_rows)"},
        {"read_vis", (PyCFunction)read_vis,
                METH_VARARGS, "read_vis(start_row, start_channel, "
                "num_channels, num_baselines, column)"},
        {"set_phase_centre", (PyCFunction)set_phase_centre,
                METH_VARARGS, "set_phase_centre(longitude_rad, latitude_rad, "
                "coord_type)"},
        {"write_coords", (PyCFunction)write_coords,
                METH_VARARGS, "write_coords(start_row, num_baselines, "
                "uu, vv, ww, exposure_sec, interval_sec, time_stamp)"},
        {"write_vis", (PyCFunction)write_vis,
                METH_VARARGS, "write_vis(start_row, start_channel, "
                "num_channels, num_baselines, vis)"},
        {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_measurement_set_lib",   /* m_name */
        module_doc,               /* m_doc */
        -1,                       /* m_size */
        methods                   /* m_methods */
};
#endif


static PyObject* moduleinit(void)
{
    PyObject* m;
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3("_measurement_set_lib", methods, module_doc);
#endif
    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__measurement_set_lib(void)
{
    import_array();
    return moduleinit();
}
#else
/* The init function name has to match that of the compiled module
 * with the pattern 'init<module name>'. This module is called '_measurement_set_lib' */
PyMODINIT_FUNC init_measurement_set_lib(void)
{
    import_array();
    moduleinit();
    return;
}
#endif

