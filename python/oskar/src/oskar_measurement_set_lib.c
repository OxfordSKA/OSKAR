/*
 * Copyright (c) 2016-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <Python.h>

#include <string.h>
#include <ms/oskar_measurement_set.h>
#include <oskar_version.h>

/* http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static const char module_doc[] =
        "This module provides an interface to the OSKAR Measurement Set wrapper.";
static const char name[] = "oskar_MeasurementSet";

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


static void ms_free(PyObject* capsule)
{
    oskar_ms_close((oskar_MeasurementSet*) get_handle(capsule, name));
}


static int numpy_type_from_ms_type(int type)
{
    switch (type)
    {
    case OSKAR_MS_BOOL:     return NPY_BOOL;
    case OSKAR_MS_CHAR:     return NPY_BYTE;
    case OSKAR_MS_UCHAR:    return NPY_UBYTE;
    case OSKAR_MS_SHORT:    return NPY_SHORT;
    case OSKAR_MS_USHORT:   return NPY_USHORT;
    case OSKAR_MS_INT:      return NPY_INT;
    case OSKAR_MS_UINT:     return NPY_UINT;
    case OSKAR_MS_FLOAT:    return NPY_FLOAT;
    case OSKAR_MS_DOUBLE:   return NPY_DOUBLE;
    case OSKAR_MS_COMPLEX:  return NPY_CFLOAT;
    case OSKAR_MS_DCOMPLEX: return NPY_CDOUBLE;
    }
    return NPY_VOID;
}


static const char* numpy_string_from_numpy_type(int type)
{
    switch (type)
    {
    case NPY_BOOL:     return "bool8";
    case NPY_BYTE:     return "byte";
    case NPY_UBYTE:    return "ubyte";
    case NPY_SHORT:    return "short";
    case NPY_USHORT:   return "ushort";
    case NPY_INT:      return "intc";
    case NPY_UINT:     return "uintc";
    case NPY_FLOAT:    return "single";
    case NPY_DOUBLE:   return "double";
    case NPY_CFLOAT:   return "complex64";
    case NPY_CDOUBLE:  return "complex128";
    }
    return "void";
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


static PyObject* column_element_size(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject* capsule = 0;
    const char* column = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &column)) return 0;
    if (!(h = (oskar_MeasurementSet*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_ms_column_element_size(h, column));
}


static PyObject* column_element_type(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject* capsule = 0;
    const char* column = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &column)) return 0;
    if (!(h = (oskar_MeasurementSet*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("s", numpy_string_from_numpy_type(
            numpy_type_from_ms_type(oskar_ms_column_element_type(h, column))));
}


static PyObject* create(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject *capsule = 0;
    int num_pols = 0, num_channels = 0, num_stations = 0;
    int write_autocorr = 0, write_crosscor = 0;
    double freq_start_hz = 0.0, freq_inc_hz = 0.0;
    const char* file_name = 0;
    if (!PyArg_ParseTuple(args, "siiiddii", &file_name, &num_stations,
            &num_channels, &num_pols, &freq_start_hz, &freq_inc_hz,
            &write_autocorr, &write_crosscor)) return 0;

    /* Create the Measurement Set. */
    h = oskar_ms_create(file_name, "Python script", num_stations, num_channels,
            num_pols, freq_start_hz, freq_inc_hz, write_autocorr,
            write_crosscor);

    /* Check for errors. */
    if (!h)
    {
        PyErr_Format(PyExc_RuntimeError,
                "Unable to create Measurement Set '%s'.", file_name);
        return 0;
    }

    /* Store the pointer in a capsule and return it. */
    capsule = PyCapsule_New((void*)h, name, (PyCapsule_Destructor)ms_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* ensure_num_rows(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject* capsule = 0;
    int num = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &num)) return 0;
    if (!(h = (oskar_MeasurementSet*) get_handle(capsule, name))) return 0;
    oskar_ms_ensure_num_rows(h, (unsigned int) num);
    return Py_BuildValue("");
}


static PyObject* freq_inc_hz(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_MeasurementSet*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_ms_freq_inc_hz(h));
}


static PyObject* freq_start_hz(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_MeasurementSet*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_ms_freq_start_hz(h));
}


static PyObject* num_channels(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_MeasurementSet*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_ms_num_channels(h));
}


static PyObject* num_pols(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_MeasurementSet*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_ms_num_pols(h));
}


static PyObject* num_rows(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_MeasurementSet*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_ms_num_rows(h));
}


static PyObject* num_stations(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_MeasurementSet*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_ms_num_stations(h));
}


static PyObject* phase_centre_ra_rad(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_MeasurementSet*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_ms_phase_centre_ra_rad(h));
}


static PyObject* phase_centre_dec_rad(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_MeasurementSet*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_ms_phase_centre_dec_rad(h));
}


static PyObject* open(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject *capsule = 0;
    const char* file_name = 0;
    PyObject *readonly;
    if (!PyArg_ParseTuple(args, "sO", &file_name, &readonly)) return 0;
#if OSKAR_VERSION > 0x020706
    if (PyObject_IsTrue(readonly))
        h = oskar_ms_open_readonly(file_name);
    else
#endif
        h = oskar_ms_open(file_name);

    /* Check for errors. */
    if (!h)
    {
        PyErr_Format(PyExc_RuntimeError,
                "Unable to open Measurement Set '%s'. "
                "Ensure it is not open elsewhere.", file_name);
        return 0;
    }

    /* Store the pointer in a capsule and return it. */
    capsule = PyCapsule_New((void*)h, name, (PyCapsule_Destructor)ms_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* read_column(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject *capsule = 0;
    size_t i = 0, ndim = 0, required_size = 0, *shape = 0;
    npy_intp *dims = 0;
    PyArrayObject* data = 0;
    int num_rows = 0, start_row = 0, status = 0, type = 0;
    const char* column = 0;
    if (!PyArg_ParseTuple(args, "Osii", &capsule, &column, &start_row,
            &num_rows))
        return 0;
    if (!(h = (oskar_MeasurementSet*) get_handle(capsule, name))) return 0;

    /* Get the type and shape of the column. */
    type = numpy_type_from_ms_type(oskar_ms_column_element_type(h, column));
    if (type == NPY_VOID)
    {
        PyErr_Format(PyExc_RuntimeError,
                "Unknown data type for column '%s'.", column);
        return 0;
    }

    /* Memory varies most rapidly with the *first* index of shape. */
    shape = oskar_ms_column_shape(h, column, &ndim);

    /* Create a numpy array to return. */
    if (!(ndim == 1 && shape[0] == 1)) ndim++;
    dims = (npy_intp*) calloc(ndim, sizeof(npy_intp));
    dims[0] = num_rows;
    for (i = 1; i < ndim; ++i) dims[i] = shape[(ndim - 2) - (i - 1)];
    data = (PyArrayObject*) PyArray_SimpleNew((int) ndim, dims, type);
    free(shape);
    free(dims);

    /* Read the data into the numpy array. */
    Py_BEGIN_ALLOW_THREADS
    oskar_ms_read_column(h, column, start_row, num_rows,
            PyArray_NBYTES(data), PyArray_DATA(data), &required_size, &status);
    Py_END_ALLOW_THREADS

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_ms_read_column() failed with code %d.", status);
        Py_XDECREF(data);
        return 0;
    }

    /* Return the data. */
    return Py_BuildValue("N", data); /* Don't increment refcount. */
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
    if (!(h = (oskar_MeasurementSet*) get_handle(capsule, name))) return 0;

    /* Create numpy arrays to return. */
    dims[0] = num_rows;
    uu = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    vv = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    ww = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    /* Read the coordinates. */
    Py_BEGIN_ALLOW_THREADS
    oskar_ms_read_coords_d(h, start_row, num_rows,
            (double*)PyArray_DATA(uu),
            (double*)PyArray_DATA(vv),
            (double*)PyArray_DATA(ww), &status);
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
    if (!(h = (oskar_MeasurementSet*) get_handle(capsule, name))) return 0;

    /* Create numpy array to return. */
    dims[0] = num_channels;
    dims[1] = num_baselines;
    dims[2] = oskar_ms_num_pols(h);
    vis = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_CFLOAT);

    /* Read the visibility data. */
    Py_BEGIN_ALLOW_THREADS
    oskar_ms_read_vis_f(h, start_row, start_channel,
            num_channels, num_baselines, column_name,
            (float*)PyArray_DATA(vis), &status);
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
    if (!(h = (oskar_MeasurementSet*) get_handle(capsule, name))) return 0;
    oskar_ms_set_phase_centre(h, 0, lon_rad, lat_rad);
    return Py_BuildValue("");
}


static PyObject* time_inc_sec(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_MeasurementSet*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_ms_time_inc_sec(h));
}


static PyObject* time_start_mjd_utc(PyObject* self, PyObject* args)
{
    oskar_MeasurementSet* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_MeasurementSet*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_ms_time_start_mjd_utc(h));
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
    if (!(h = (oskar_MeasurementSet*) get_handle(capsule, name))) return 0;

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

    /* Write the coordinates. */
    Py_BEGIN_ALLOW_THREADS
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
    if (!(h = (oskar_MeasurementSet*) get_handle(capsule, name))) return 0;

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

    /* Write the visibilities. */
    Py_BEGIN_ALLOW_THREADS
    if (PyArray_TYPE(vis) == NPY_DOUBLE)
        oskar_ms_write_vis_d(h, start_row, start_channel, num_channels,
                num_baselines, (const double*)PyArray_DATA(vis));
    else
        oskar_ms_write_vis_f(h, start_row, start_channel, num_channels,
                num_baselines, (const float*)PyArray_DATA(vis));
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
        {"capsule_name", (PyCFunction)capsule_name,
                METH_VARARGS, "capsule_name()"},
        {"column_element_type", (PyCFunction)column_element_type,
                METH_VARARGS, "column_element_type(column)"},
        {"column_element_size", (PyCFunction)column_element_size,
                METH_VARARGS, "column_element_size(column)"},
        {"create", (PyCFunction)create,
                METH_VARARGS, "create(file_name, num_stations, num_channels, "
                "num_pols, ref_freq_hz, chan_width_hz, "
                "write_autocorr, write_crosscor)"},
        {"ensure_num_rows", (PyCFunction)ensure_num_rows,
                METH_VARARGS, "ensure_num_rows(num_rows)"},
        {"freq_inc_hz", (PyCFunction)freq_inc_hz,
                METH_VARARGS, "freq_inc_hz()"},
        {"freq_start_hz", (PyCFunction)freq_start_hz,
                METH_VARARGS, "freq_start_hz()"},
        {"num_channels", (PyCFunction)num_channels,
                METH_VARARGS, "num_channels()"},
        {"num_pols", (PyCFunction)num_pols, METH_VARARGS, "num_pols()"},
        {"num_rows", (PyCFunction)num_rows, METH_VARARGS, "num_rows()"},
        {"num_stations", (PyCFunction)num_stations,
                METH_VARARGS, "num_stations()"},
        {"open", (PyCFunction)open, METH_VARARGS, "open(filename)"},
        {"phase_centre_ra_rad", (PyCFunction)phase_centre_ra_rad,
                METH_VARARGS, "phase_centre_ra_rad()"},
        {"phase_centre_dec_rad", (PyCFunction)phase_centre_dec_rad,
                METH_VARARGS, "phase_centre_dec_rad()"},
        {"read_column", (PyCFunction)read_column,
                METH_VARARGS, "read_column(column, start_row, num_rows)"},
        {"read_coords", (PyCFunction)read_coords,
                METH_VARARGS, "read_coords(start_row, num_rows)"},
        {"read_vis", (PyCFunction)read_vis,
                METH_VARARGS, "read_vis(start_row, start_channel, "
                "num_channels, num_baselines, column)"},
        {"set_phase_centre", (PyCFunction)set_phase_centre,
                METH_VARARGS, "set_phase_centre(longitude_rad, latitude_rad, "
                "coord_type)"},
        {"time_inc_sec", (PyCFunction)time_inc_sec,
                METH_VARARGS, "time_inc_sec()"},
        {"time_start_mjd_utc", (PyCFunction)time_start_mjd_utc,
                METH_VARARGS, "time_start_mjd_utc()"},
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

