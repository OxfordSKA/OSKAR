/*
 * Copyright (c) 2014, The University of Oxford
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
//#include <numpy/arrayobject.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <oskar_global.h>
#include <oskar_mem.h>

void mem_free(void* ptr)
{
    //oskar_mem_free(
    // TODO: Py_DECREF(..)
    printf("PyCapsule destructor for oskar_Mem called!\n");
}

static PyObject* create(PyObject* self, PyObject* args, PyObject* keywds)
{
    int type_ = OSKAR_DOUBLE;
    int loc_ = OSKAR_CPU;
    int length_ = 0;

    static char* keywords[] = {"type", "location", "length", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "iii", keywords, &type_,
            &loc_, &length_))
        return NULL;

    int status = OSKAR_SUCCESS;
    oskar_Mem* mem = oskar_mem_create(type_, loc_, length_, &status);

    const char* name = "oskar_Mem";
    PyObject* mem_ =  PyCapsule_New((void*)mem,name,(PyCapsule_Destructor)mem_free);
    return Py_BuildValue("Oi", mem_, status);
}

static PyObject* get_location(PyObject* self, PyObject* args)
{
    PyObject* mem_ = NULL;
    if (!PyArg_ParseTuple(args, "O", &mem_))
        return NULL;

    if (!PyCapsule_CheckExact(mem_)) {
        printf("Input argument not a PyCapsule object!\n");
        return NULL;
    }

    oskar_Mem* mem = (oskar_Mem*)PyCapsule_GetPointer(mem_, "oskar_Mem");
    if (!mem) {
        printf("Unable to convert PyCapsule object to pointer :(\n");
        return NULL;
    }

    return Py_BuildValue("i", oskar_mem_location(mem));
}

// Methods table
static PyMethodDef oskar_mem_lib_methods[] =
{
    {"create", (PyCFunction)create, METH_VARARGS | METH_KEYWORDS,
    "create(...)"},
    {"location", (PyCFunction)get_location, METH_VARARGS,
    "location(mem)"},
    {NULL, NULL, 0, NULL}
};

// Initialisation function (called init[filename] where filename = name of *.so)
// http://docs.python.org/2/extending/extending.html
PyMODINIT_FUNC init_mem_lib(void)
{
    (void) Py_InitModule3("_mem_lib", oskar_mem_lib_methods, "docstring...");
    // Import the use of numpy array objects.
    //import_array();
}
