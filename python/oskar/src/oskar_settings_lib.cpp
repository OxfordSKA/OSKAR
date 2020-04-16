/*
 * Copyright (c) 2017-2019, The University of Oxford
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

#include <stdio.h>
#include <stdlib.h>

#include <Python.h>

#include <oskar.h>

/*
 * NOTE: Don't use std::string here!
 * It may not be ABI-compatible with the version in the C++ library
 * linked by OSKAR.
 */

using oskar::SettingsTree;

static const char module_doc[] =
        "This module provides an interface to the OSKAR settings tree.";
static const char name[] = "oskar_SettingsTree";

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

static void settings_free(PyObject* capsule)
{
    SettingsTree* h = (SettingsTree*) get_handle(capsule, name);
    SettingsTree::free(h);
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
    (void) self;
    (void) args;
    SettingsTree* h = new SettingsTree;
    PyObject *capsule = PyCapsule_New((void*)h, name,
            (PyCapsule_Destructor)settings_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* iterate_dict(PyObject* dict,
        SettingsTree* h, PyObject* parent_key, bool& abort)
{
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(dict, &pos, &key, &value) && !abort)
    {
#if PY_MAJOR_VERSION >= 3
        if (!PyUnicode_Check(key))
#else
        if (!PyString_Check(key))
#endif
        {
            PyErr_SetString(PyExc_RuntimeError,
                    "Dictionary keys must be strings.");
            abort = true;
            return 0;
        }
        if (PyDict_Check(value))
            (void) iterate_dict(value, h, key, abort);
        else
        {
            PyObject* pystr;
            const char *k, *v, *parent = 0, *key_ptr = 0;
            char *full_key = 0;

            /* Get the fully-qualified key. */
#if PY_MAJOR_VERSION >= 3
            k = PyUnicode_AsUTF8(key);
            if (parent_key)
                parent = PyUnicode_AsUTF8(parent_key);
#else
            k = PyString_AsString(key);
            if (parent_key)
                parent = PyString_AsString(parent_key);
#endif
            if (!parent)
                key_ptr = k;
            else
            {
                full_key = (char*) calloc(2 + strlen(parent) + strlen(k), 1);
                key_ptr = full_key;
                sprintf(full_key, "%s%c%s", parent, h->separator(), k);
            }

            /* Try to convert the value to a string. */
            pystr = PyObject_Str(value);
            if (!abort && !pystr)
            {
                PyErr_Format(PyExc_RuntimeError,
                        "Could not convert value for '%s' to a Python string",
                        key_ptr);
                abort = true;
            }
            else
            {
#if PY_MAJOR_VERSION >= 3
                v = PyUnicode_AsUTF8(pystr);
#else
                v = PyString_AsString(pystr);
#endif
            }
            if (!abort && !v)
            {
                PyErr_Format(PyExc_RuntimeError,
                        "Could not convert value for '%s' to a C string",
                        key_ptr);
                abort = true;
            }

            /* Try to set the value. */
            if (!abort && !h->set_value(key_ptr, v))
            {
                PyErr_Format(PyExc_RuntimeError,
                        "Could not set '%s'='%s'", key_ptr, v);
                abort = true;
            }
            free(full_key);
            Py_XDECREF(pystr);
            if (abort) return 0;
        }
    }
    return abort ? 0 : Py_BuildValue("");
}

static PyObject* from_dict(PyObject* self, PyObject* args)
{
    SettingsTree* h = 0;
    PyObject* obj = 0, *dict = 0;
    if (!PyArg_ParseTuple(args, "OO", &obj, &dict)) return 0;
    if (!(h = (SettingsTree*) get_handle(obj, name))) return 0;

    /* Iterate the dictionary. */
    bool abort = false;
    return iterate_dict(dict, h, 0, abort);
}


static PyObject* set_value(PyObject* self, PyObject* args)
{
    SettingsTree* h = 0;
    PyObject* obj = 0;
    const char *key = 0, *value = 0;
    int write = 1;
    if (!PyArg_ParseTuple(args, "Ossi", &obj, &key, &value, &write)) return 0;
    if (!(h = (SettingsTree*) get_handle(obj, name))) return 0;

    /* Set the value of the key. */
    if (!h->set_value(key, value, (bool) write))
    {
        PyErr_Format(PyExc_RuntimeError,
                "Unable to set '%s'='%s'.", key, value);
        return 0;
    }
    return Py_BuildValue("");
}


static void iterate_nodes(const oskar::SettingsNode* node,
        PyObject* dict, int include_defaults)
{
    if (node->item_type() == oskar::SettingsItem::SETTING &&
            (node->is_set() || include_defaults))
    {
        PyObject* str;
#if PY_MAJOR_VERSION >= 3
        str = PyUnicode_FromString(node->value());
#else
        str = PyString_FromString(node->value());
#endif
        PyDict_SetItemString(dict, node->key(), str);
    }
    for (int i = 0; i < node->num_children(); ++i)
        iterate_nodes(node->child(i), dict, include_defaults);
}

static PyObject* to_dict(PyObject* self, PyObject* args)
{
    SettingsTree* h = 0;
    PyObject* obj = 0, *dict = 0;
    int include_defaults = 0;
    if (!PyArg_ParseTuple(args, "Oi", &obj, &include_defaults)) return 0;
    if (!(h = (SettingsTree*) get_handle(obj, name))) return 0;

    /* Iterate the tree from the root node. */
    dict = PyDict_New();
    iterate_nodes(h->root_node(), dict, include_defaults);
    return Py_BuildValue("N", dict); /* Don't increment refcount. */
}


static PyObject* value(PyObject* self, PyObject* args)
{
    SettingsTree* h = 0;
    PyObject* obj = 0;
    const char* key = 0;
    if (!PyArg_ParseTuple(args, "Os", &obj, &key)) return 0;
    if (!(h = (SettingsTree*) get_handle(obj, name))) return 0;

    /* Return the value of the key. */
    int status = 0;
    const char* value = h->to_string(key, &status);
    if (status || !value)
    {
        PyErr_Format(PyExc_KeyError, "Unable to return value for '%s'.", key);
        return 0;
    }
    return Py_BuildValue("s", value);
}


/* Method table. */
static PyMethodDef methods[] =
{
        {"capsule_name", (PyCFunction)capsule_name,
                METH_VARARGS, "capsule_name()"},
        {"create", (PyCFunction)create, METH_VARARGS, "create()"},
        {"from_dict", (PyCFunction)from_dict, METH_VARARGS, "from_dict(dict)"},
        {"set_value", (PyCFunction)set_value,
                METH_VARARGS, "set_value(key, value)"},
        {"to_dict", (PyCFunction)to_dict, METH_VARARGS, "to_dict(dict)"},
        {"value", (PyCFunction)value, METH_VARARGS, "value(key)"},
        {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_settings_lib",    /* m_name */
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
    m = Py_InitModule3("_settings_lib", methods, module_doc);
#endif
    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__settings_lib(void)
{
    return moduleinit();
}
#else
/* The init function name has to match that of the compiled module
 * with the pattern 'init<module name>'. This module is called '_settings_lib' */
PyMODINIT_FUNC init_settings_lib(void)
{
    moduleinit();
    return;
}
#endif

