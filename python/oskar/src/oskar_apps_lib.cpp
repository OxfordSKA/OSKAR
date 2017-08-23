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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * NOTE: Don't use std::string here!
 * It may not be ABI-compatible with the version in the C++ library
 * linked by OSKAR.
 */

using oskar::SettingsTree;

static const char module_doc[] =
        "This module provides an interface to the OSKAR application settings.";
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

static void beam_pattern_free(PyObject* capsule)
{
    int status = 0;
    oskar_beam_pattern_free((oskar_BeamPattern*) get_handle(capsule,
            "oskar_BeamPattern"), &status);
}

static void imager_free(PyObject* capsule)
{
    int status = 0;
    oskar_imager_free((oskar_Imager*) get_handle(capsule,
            "oskar_Imager"), &status);
}

static void settings_free(PyObject* capsule)
{
    SettingsTree* h = (SettingsTree*) get_handle(capsule, name);
    SettingsTree::free(h);
}

static void interferometer_free(PyObject* capsule)
{
    int status = 0;
    oskar_interferometer_free((oskar_Interferometer*) get_handle(capsule,
            "oskar_Interferometer"), &status);
}

static void sky_free(PyObject* capsule)
{
    int status = 0;
    oskar_sky_free((oskar_Sky*) get_handle(capsule, "oskar_Sky"), &status);
}

static void telescope_free(PyObject* capsule)
{
    int status = 0;
    oskar_telescope_free((oskar_Telescope*) get_handle(capsule,
            "oskar_Telescope"), &status);
}


static PyObject* settings_to_beam_pattern(PyObject* self, PyObject* args)
{
    SettingsTree* h = 0;
    PyObject* obj = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &obj)) return 0;
    if (!(h = (SettingsTree*) get_handle(obj, name))) return 0;
    oskar_BeamPattern* sim = oskar_settings_to_beam_pattern(h, 0, &status);
    if (!sim || status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_settings_to_beam_pattern() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    PyObject *capsule = PyCapsule_New((void*)sim, "oskar_BeamPattern",
            (PyCapsule_Destructor)beam_pattern_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* settings_to_imager(PyObject* self, PyObject* args)
{
    SettingsTree* h = 0;
    PyObject* obj = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &obj)) return 0;
    if (!(h = (SettingsTree*) get_handle(obj, name))) return 0;
    oskar_Imager* im = oskar_settings_to_imager(h, 0, &status);
    if (!im || status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_settings_to_imager() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    PyObject *capsule = PyCapsule_New((void*)im, "oskar_Imager",
            (PyCapsule_Destructor)imager_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* settings_to_interferometer(PyObject* self, PyObject* args)
{
    SettingsTree* h = 0;
    PyObject* obj = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &obj)) return 0;
    if (!(h = (SettingsTree*) get_handle(obj, name))) return 0;
    oskar_Interferometer* sim = oskar_settings_to_interferometer(h, 0, &status);
    if (!sim || status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_settings_to_interferometer() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    PyObject *capsule = PyCapsule_New((void*)sim, "oskar_Interferometer",
            (PyCapsule_Destructor)interferometer_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* settings_to_sky(PyObject* self, PyObject* args)
{
    SettingsTree* h = 0;
    PyObject* obj = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &obj)) return 0;
    if (!(h = (SettingsTree*) get_handle(obj, name))) return 0;
    oskar_Sky* sky = oskar_settings_to_sky(h, 0, &status);
    if (!sky || status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_settings_to_sky() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    PyObject *capsule = PyCapsule_New((void*)sky, "oskar_Sky",
            (PyCapsule_Destructor)sky_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* settings_to_telescope(PyObject* self, PyObject* args)
{
    SettingsTree* h = 0;
    PyObject* obj = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &obj)) return 0;
    if (!(h = (SettingsTree*) get_handle(obj, name))) return 0;
    oskar_Telescope* tel = oskar_settings_to_telescope(h, 0, &status);
    if (!tel || status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_settings_to_telescope() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    PyObject *capsule = PyCapsule_New((void*)tel, "oskar_Telescope",
            (PyCapsule_Destructor)telescope_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* settings_tree(PyObject* self, PyObject* args)
{
    const char *app = 0, *settings_file = 0;
    if (!PyArg_ParseTuple(args, "ss", &app, &settings_file)) return 0;

    SettingsTree* h = oskar_app_settings_tree(app, settings_file);
    if (!h)
    {
        if (strlen(settings_file) > 0)
            PyErr_Format(PyExc_RuntimeError,
                    "Failed to read settings file '%s'.", settings_file);
        else
            PyErr_SetString(PyExc_RuntimeError,
                    "oskar_app_settings_tree() failed.");
        return 0;
    }
    if (h->num_failed_keys() > 0)
    {
        char* message = 0;
        for (int i = 0; i < h->num_failed_keys(); ++i)
        {
            message = (char*) realloc(message, 20 +
                    strlen(h->failed_key(i)) + strlen(h->failed_key_value(i)));
            sprintf(message, "Ignoring '%s'='%s'",
                    h->failed_key(i), h->failed_key_value(i));
            PyErr_WarnEx(PyExc_SyntaxWarning, message, 2);
        }
        free(message);
    }

    PyObject *capsule = PyCapsule_New((void*)h, name,
            (PyCapsule_Destructor)settings_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


/* Method table. */
static PyMethodDef methods[] =
{
        {"settings_to_beam_pattern", (PyCFunction)settings_to_beam_pattern,
                METH_VARARGS, "settings_to_beam_pattern(settings_tree)"},
        {"settings_to_imager", (PyCFunction)settings_to_imager,
                METH_VARARGS, "settings_to_imager(settings_tree)"},
        {"settings_to_interferometer", (PyCFunction)settings_to_interferometer,
                METH_VARARGS, "settings_to_interferometer(settings_tree)"},
        {"settings_to_sky", (PyCFunction)settings_to_sky,
                METH_VARARGS, "settings_to_sky(settings_tree)"},
        {"settings_to_telescope", (PyCFunction)settings_to_telescope,
                METH_VARARGS, "settings_to_telescope(settings_tree)"},
        {"settings_tree", (PyCFunction)settings_tree,
                METH_VARARGS, "settings_tree(app_name, settings_file)"},
        {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_apps_lib",        /* m_name */
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
    m = Py_InitModule3("_apps_lib", methods, module_doc);
#endif
    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__apps_lib(void)
{
    return moduleinit();
}
#else
/* The init function name has to match that of the compiled module
 * with the pattern 'init<module name>'. This module is called '_apps_lib' */
PyMODINIT_FUNC init_apps_lib(void)
{
    moduleinit();
    return;
}
#endif

