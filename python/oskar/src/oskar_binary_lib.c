/*
 * Copyright (c) 2017, The University of Oxford
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
#include <stdlib.h>

/* http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static const char module_doc[] =
        "This module provides an interface to an OSKAR binary file.";
static const char name[] = "oskar_Binary";

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


static void binary_free(PyObject* capsule)
{
    oskar_binary_free((oskar_Binary*) get_handle(capsule, name));
}


static int numpy_type_from_oskar(int type)
{
    switch (type)
    {
    case OSKAR_INT:                   return NPY_INT;
    case OSKAR_SINGLE:                return NPY_FLOAT;
    case OSKAR_DOUBLE:                return NPY_DOUBLE;
    case OSKAR_SINGLE_COMPLEX:        return NPY_CFLOAT;
    case OSKAR_DOUBLE_COMPLEX:        return NPY_CDOUBLE;
    case OSKAR_SINGLE_COMPLEX_MATRIX: return NPY_CFLOAT;
    case OSKAR_DOUBLE_COMPLEX_MATRIX: return NPY_CDOUBLE;
    }
    return 0;
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


static PyObject* create(PyObject* self, PyObject* args)
{
    oskar_Binary* h = 0;
    PyObject *capsule = 0;
    int status = 0;
    const char *filename = 0;
    char mode = 0;
    if (!PyArg_ParseTuple(args, "sc", &filename, &mode)) return 0;
    h = oskar_binary_create(filename, mode, &status);

    /* Check for errors. */
    if (!h || status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_binary_create() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        oskar_binary_free(h);
        return 0;
    }

    /* Store the handle in the PyCapsule and return it. */
    capsule = PyCapsule_New((void*)h, name,
            (PyCapsule_Destructor)binary_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* num_tags(PyObject* self, PyObject* args)
{
    oskar_Binary* h = 0;
    PyObject *capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Binary*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_binary_num_tags(h));
}


static PyObject* read_data(PyObject* self, PyObject* args)
{
    oskar_Binary* h = 0;
    PyObject *capsule = 0, *grp = 0, *tag = 0, *data_type = 0, *array = 0;
    int dtype = 0, i = 0, tag_idx = 0, status = 0, use_ints = 0, use_strs = 0;
    size_t num_elements = 0, bytes = 0;
    if (!PyArg_ParseTuple(args, "OOOiO", &capsule, &grp, &tag, &tag_idx,
            &data_type))
        return 0;
    if (!(h = (oskar_Binary*) get_handle(capsule, name))) return 0;

    /* Get requested data type, if supplied. */
    if (data_type != Py_None)
    {
#if PY_MAJOR_VERSION >= 3
        dtype = (int) PyLong_AsLong(data_type);
#else
        dtype = (int) PyInt_AsLong(data_type);
#endif
    }

    /* Check if group and tag identifiers are integers or strings
     * and get the chunk index, accordingly. */
#if PY_MAJOR_VERSION >= 3
    use_ints = PyLong_Check(grp) && PyLong_Check(tag);
    use_strs = PyUnicode_Check(grp) && PyUnicode_Check(tag);
#else
    use_ints = PyInt_Check(grp) && PyInt_Check(tag);
    use_strs = PyString_Check(grp) && PyString_Check(tag);
#endif
    if (use_ints)
    {
        int i_grp, i_tag;
#if PY_MAJOR_VERSION >= 3
        i_grp = (int) PyLong_AsLong(grp);
        i_tag = (int) PyLong_AsLong(tag);
#else
        i_grp = (int) PyInt_AsLong(grp);
        i_tag = (int) PyInt_AsLong(tag);
#endif
        i = oskar_binary_query(h, (unsigned char) dtype, (unsigned char) i_grp,
                (unsigned char) i_tag, tag_idx, &bytes, &status);
        if (i < 0)
        {
            PyErr_Format(PyExc_RuntimeError,
                    "Tag (%i,%i:%i) not found in binary file.",
                    i_grp, i_tag, tag_idx);
            return 0;
        }
    }
    else if (use_strs)
    {
        char *s_grp, *s_tag;
#if PY_MAJOR_VERSION >= 3
        s_grp = PyUnicode_AsUTF8(grp);
        s_tag = PyUnicode_AsUTF8(tag);
#else
        s_grp = PyString_AsString(grp);
        s_tag = PyString_AsString(tag);
#endif
        i = oskar_binary_query_ext(h, (unsigned char) dtype,
                s_grp, s_tag, tag_idx, &bytes, &status);
        if (i < 0)
        {
            PyErr_Format(PyExc_RuntimeError,
                    "Tag (%s,%s:%i) not found in binary file.",
                    s_grp, s_tag, tag_idx);
            return 0;
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "Unknown binary tag specifiers.");
        return 0;
    }

    /* Get the actual data type and number of elements in the chunk. */
    dtype = oskar_binary_tag_data_type(h, i);
    num_elements = bytes / oskar_mem_element_size(dtype);

    /* Read the data chunk and return the appropriate object to Python. */
    if (dtype == OSKAR_CHAR)
    {
        /* Read a string. */
        char* data = (char*) calloc(bytes, 1);
        oskar_binary_read_block(h, i, bytes, data, &status);
        if (status) goto fail;
        while (bytes > 1 && data[bytes - 1] == '\0') bytes--;
#if PY_MAJOR_VERSION >= 3
        array = PyUnicode_DecodeUTF8(data, bytes, NULL);
#else
        array = PyString_FromStringAndSize(data, bytes);
#endif
        free(data);
        return Py_BuildValue("N", array);
    }
    else if ((dtype == OSKAR_INT || dtype == OSKAR_SINGLE ||
            dtype == OSKAR_DOUBLE) && num_elements == 1)
    {
        /* Read a single value. */
        switch (dtype)
        {
        case OSKAR_INT:
        {
            int val;
            oskar_binary_read_block(h, i, sizeof(int), &val, &status);
            if (status) goto fail;
            return Py_BuildValue("i", val);
        }
        case OSKAR_SINGLE:
        {
            float val;
            oskar_binary_read_block(h, i, sizeof(float), &val, &status);
            if (status) goto fail;
            return Py_BuildValue("f", val);
        }
        case OSKAR_DOUBLE:
        {
            double val;
            oskar_binary_read_block(h, i, sizeof(double), &val, &status);
            if (status) goto fail;
            return Py_BuildValue("d", val);
        }
        }
    }
    else
    {
        /* Read an array. */
        npy_intp dims[3] = {0, 2, 2};
        dims[0] = num_elements;
        array = PyArray_SimpleNew(oskar_type_is_matrix(dtype) ? 3 : 1,
                dims, numpy_type_from_oskar(dtype));
        Py_BEGIN_ALLOW_THREADS
        oskar_binary_read_block(h, i, PyArray_NBYTES((PyArrayObject*)array),
                PyArray_DATA((PyArrayObject*)array), &status);
        Py_END_ALLOW_THREADS
        if (status) goto fail;
        return Py_BuildValue("N", array);
    }

fail:
    Py_XDECREF(array);
    PyErr_Format(PyExc_RuntimeError,
            "oskar_binary_read_block() failed with code %d (%s).",
        status, oskar_get_error_string(status));
    return 0;
}


static PyObject* set_query_search_start(PyObject* self, PyObject* args)
{
    oskar_Binary* h = 0;
    PyObject *capsule = 0;
    int start = 0, status = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &start)) return 0;
    if (!(h = (oskar_Binary*) get_handle(capsule, name))) return 0;
    oskar_binary_set_query_search_start(h, start, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_binary_set_query_search_start() failed "
                "with code %d (%s).", status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


/* Method table. */
static PyMethodDef methods[] =
{
        {"capsule_name", (PyCFunction)capsule_name,
                METH_VARARGS, "capsule_name()"},
        {"create", (PyCFunction)create,
                METH_VARARGS, "create(filename, mode)"},
        {"num_tags", (PyCFunction)num_tags, METH_VARARGS, "num_tags()"},
        {"read_data", (PyCFunction)read_data,
                METH_VARARGS, "read_data(group, tag, user_index, data_type)"},
        {"set_query_search_start", (PyCFunction)set_query_search_start,
                METH_VARARGS, "set_query_search_start(start)"},
        {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_binary_lib",      /* m_name */
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
    m = Py_InitModule3("_binary_lib", methods, module_doc);
#endif
    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__binary_lib(void)
{
    import_array();
    return moduleinit();
}
#else
/* The init function name has to match that of the compiled module
 * with the pattern 'init<module name>'. This module is called '_binary_lib' */
PyMODINIT_FUNC init_binary_lib(void)
{
    import_array();
    moduleinit();
    return;
}
#endif

