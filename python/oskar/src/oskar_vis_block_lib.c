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

#include <oskar.h>
#include <string.h>

/* http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static const char module_doc[] =
        "This module provides an interface to an OSKAR visibility block.";
static const char name[] = "oskar_VisBlock";

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


static void vis_block_free(PyObject* capsule)
{
    int status = 0;
    oskar_vis_block_free((oskar_VisBlock*) get_handle(capsule, name), &status);
}


static PyObject* auto_correlations(PyObject* self, PyObject* args)
{
    oskar_VisBlock* h = 0;
    oskar_Mem* m = 0;
    PyObject *capsule = 0;
    PyArrayObject *array = 0;
    npy_intp dims[4];
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisBlock*) get_handle(capsule, name))) return 0;

    /* Check that auto-correlations exist. */
    if (!oskar_vis_block_has_auto_correlations(h))
    {
        PyErr_SetString(PyExc_RuntimeError, "No auto-correlations in block.");
        return 0;
    }

    /* Return an array reference to Python. */
    m = oskar_vis_block_auto_correlations(h);
    dims[0] = oskar_vis_block_num_times(h);
    dims[1] = oskar_vis_block_num_channels(h);
    dims[2] = oskar_vis_block_num_stations(h);
    dims[3] = oskar_vis_block_num_pols(h);
    array = (PyArrayObject*)PyArray_SimpleNewFromData(4, dims,
            (oskar_mem_is_double(m) ? NPY_CDOUBLE : NPY_CFLOAT),
            oskar_mem_void(m));
    return Py_BuildValue("N", array); /* Don't increment refcount. */
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


static PyObject* create_from_header(PyObject* self, PyObject* args)
{
    oskar_VisBlock* h = 0;
    oskar_VisHeader* hdr = 0;
    PyObject *capsule = 0, *header = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &header)) return 0;
    if (!(hdr = (oskar_VisHeader*) get_handle(header, "oskar_VisHeader")))
        return 0;
    h = oskar_vis_block_create_from_header(OSKAR_CPU, hdr, &status);
    capsule = PyCapsule_New((void*)h, name,
            (PyCapsule_Destructor)vis_block_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* baseline_uu_metres(PyObject* self, PyObject* args)
{
    oskar_VisBlock* h = 0;
    oskar_Mem* m = 0;
    PyObject *capsule = 0;
    PyArrayObject *array = 0;
    npy_intp dims[2];
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisBlock*) get_handle(capsule, name))) return 0;

    /* Check that cross-correlations exist. */
    if (!oskar_vis_block_has_cross_correlations(h))
    {
        PyErr_SetString(PyExc_RuntimeError, "No cross-correlations in block.");
        return 0;
    }

    /* Return an array reference to Python. */
    m = oskar_vis_block_baseline_uu_metres(h);
    dims[0] = oskar_vis_block_num_times(h);
    dims[1] = oskar_vis_block_num_baselines(h);
    array = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims,
            (oskar_mem_is_double(m) ? NPY_DOUBLE : NPY_FLOAT),
            oskar_mem_void(m));
    return Py_BuildValue("N", array); /* Don't increment refcount. */
}


static PyObject* baseline_vv_metres(PyObject* self, PyObject* args)
{
    oskar_VisBlock* h = 0;
    oskar_Mem* m = 0;
    PyObject *capsule = 0;
    PyArrayObject *array = 0;
    npy_intp dims[2];
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisBlock*) get_handle(capsule, name))) return 0;

    /* Check that cross-correlations exist. */
    if (!oskar_vis_block_has_cross_correlations(h))
    {
        PyErr_SetString(PyExc_RuntimeError, "No cross-correlations in block.");
        return 0;
    }

    /* Return an array reference to Python. */
    m = oskar_vis_block_baseline_vv_metres(h);
    dims[0] = oskar_vis_block_num_times(h);
    dims[1] = oskar_vis_block_num_baselines(h);
    array = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims,
            (oskar_mem_is_double(m) ? NPY_DOUBLE : NPY_FLOAT),
            oskar_mem_void(m));
    return Py_BuildValue("N", array); /* Don't increment refcount. */
}


static PyObject* baseline_ww_metres(PyObject* self, PyObject* args)
{
    oskar_VisBlock* h = 0;
    oskar_Mem* m = 0;
    PyObject *capsule = 0;
    PyArrayObject *array = 0;
    npy_intp dims[2];
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisBlock*) get_handle(capsule, name))) return 0;

    /* Check that cross-correlations exist. */
    if (!oskar_vis_block_has_cross_correlations(h))
    {
        PyErr_SetString(PyExc_RuntimeError, "No cross-correlations in block.");
        return 0;
    }

    /* Return an array reference to Python. */
    m = oskar_vis_block_baseline_ww_metres(h);
    dims[0] = oskar_vis_block_num_times(h);
    dims[1] = oskar_vis_block_num_baselines(h);
    array = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims,
            (oskar_mem_is_double(m) ? NPY_DOUBLE : NPY_FLOAT),
            oskar_mem_void(m));
    return Py_BuildValue("N", array); /* Don't increment refcount. */
}


static PyObject* cross_correlations(PyObject* self, PyObject* args)
{
    oskar_VisBlock* h = 0;
    oskar_Mem* m = 0;
    PyObject *capsule = 0;
    PyArrayObject *array = 0;
    npy_intp dims[4];
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisBlock*) get_handle(capsule, name))) return 0;

    /* Check that cross-correlations exist. */
    if (!oskar_vis_block_has_cross_correlations(h))
    {
        PyErr_SetString(PyExc_RuntimeError, "No cross-correlations in block.");
        return 0;
    }

    /* Return an array reference to Python. */
    m = oskar_vis_block_cross_correlations(h);
    dims[0] = oskar_vis_block_num_times(h);
    dims[1] = oskar_vis_block_num_channels(h);
    dims[2] = oskar_vis_block_num_baselines(h);
    dims[3] = oskar_vis_block_num_pols(h);
    array = (PyArrayObject*)PyArray_SimpleNewFromData(4, dims,
            (oskar_mem_is_double(m) ? NPY_CDOUBLE : NPY_CFLOAT),
            oskar_mem_void(m));
    return Py_BuildValue("N", array); /* Don't increment refcount. */
}


static PyObject* num_baselines(PyObject* self, PyObject* args)
{
    oskar_VisBlock* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisBlock*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_vis_block_num_baselines(h));
}


static PyObject* num_channels(PyObject* self, PyObject* args)
{
    oskar_VisBlock* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisBlock*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_vis_block_num_channels(h));
}


static PyObject* num_pols(PyObject* self, PyObject* args)
{
    oskar_VisBlock* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisBlock*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_vis_block_num_pols(h));
}


static PyObject* num_stations(PyObject* self, PyObject* args)
{
    oskar_VisBlock* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisBlock*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_vis_block_num_stations(h));
}


static PyObject* num_times(PyObject* self, PyObject* args)
{
    oskar_VisBlock* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisBlock*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_vis_block_num_times(h));
}


static PyObject* start_channel_index(PyObject* self, PyObject* args)
{
    oskar_VisBlock* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisBlock*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_vis_block_start_channel_index(h));
}


static PyObject* start_time_index(PyObject* self, PyObject* args)
{
    oskar_VisBlock* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_VisBlock*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_vis_block_start_time_index(h));
}


/* Method table. */
static PyMethodDef methods[] =
{
        {"auto_correlations", (PyCFunction)auto_correlations,
                METH_VARARGS, "auto_correlations()"},
        {"capsule_name", (PyCFunction)capsule_name,
                METH_VARARGS, "capsule_name()"},
        {"create_from_header", (PyCFunction)create_from_header,
                METH_VARARGS, "create_from_header(header)"},
        {"baseline_uu_metres", (PyCFunction)baseline_uu_metres,
                METH_VARARGS, "baseline_uu_metres()"},
        {"baseline_vv_metres", (PyCFunction)baseline_vv_metres,
                METH_VARARGS, "baseline_vv_metres()"},
        {"baseline_ww_metres", (PyCFunction)baseline_ww_metres,
                METH_VARARGS, "baseline_ww_metres()"},
        {"cross_correlations", (PyCFunction)cross_correlations,
                METH_VARARGS, "cross_correlations()"},
        {"num_baselines", (PyCFunction)num_baselines,
                METH_VARARGS, "num_baselines()"},
        {"num_channels", (PyCFunction)num_channels,
                METH_VARARGS, "num_channels()"},
        {"num_pols", (PyCFunction)num_pols,
                METH_VARARGS, "num_pols()"},
        {"num_stations", (PyCFunction)num_stations,
                METH_VARARGS, "num_stations()"},
        {"num_times", (PyCFunction)num_times,
                METH_VARARGS, "num_times()"},
        {"start_channel_index", (PyCFunction)start_channel_index,
                METH_VARARGS, "start_channel_index()"},
        {"start_time_index", (PyCFunction)start_time_index,
                METH_VARARGS, "start_time_index()"},
        {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_vis_block_lib",   /* m_name */
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
    m = Py_InitModule3("_vis_block_lib", methods, module_doc);
#endif
    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__vis_block_lib(void)
{
    import_array();
    return moduleinit();
}
#else
/* The init function name has to match that of the compiled module
 * with the pattern 'init<module name>'. This module is called '_vis_block_lib' */
PyMODINIT_FUNC init_vis_block_lib(void)
{
    import_array();
    moduleinit();
    return;
}
#endif

