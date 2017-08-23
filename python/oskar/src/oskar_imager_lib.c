/*
 * Copyright (c) 2014-2017, The University of Oxford
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
#include <stdlib.h>
#include <string.h>

/* http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static const char module_doc[] =
        "This module provides an interface to the OSKAR imager.";
static const char name[] = "oskar_Imager";

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


static void imager_free(PyObject* capsule)
{
    int status = 0;
    oskar_imager_free((oskar_Imager*) get_handle(capsule, name), &status);
}


static int numpy_type_from_oskar(int type)
{
    switch (type)
    {
    case OSKAR_INT:            return NPY_INT;
    case OSKAR_SINGLE:         return NPY_FLOAT;
    case OSKAR_DOUBLE:         return NPY_DOUBLE;
    case OSKAR_SINGLE_COMPLEX: return NPY_CFLOAT;
    case OSKAR_DOUBLE_COMPLEX: return NPY_CDOUBLE;
    }
    return 0;
}


static int oskar_type_from_numpy(PyArrayObject* arr)
{
    switch (PyArray_TYPE(arr))
    {
    case NPY_INT:     return OSKAR_INT;
    case NPY_FLOAT:   return OSKAR_SINGLE;
    case NPY_DOUBLE:  return OSKAR_DOUBLE;
    case NPY_CFLOAT:  return OSKAR_SINGLE_COMPLEX;
    case NPY_CDOUBLE: return OSKAR_DOUBLE_COMPLEX;
    }
    return 0;
}


static PyObject* algorithm(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("s", oskar_imager_algorithm(h));
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


static PyObject* cellsize(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_imager_cellsize(h));
}


static PyObject* channel_snapshots(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int flag = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    flag = oskar_imager_channel_snapshots(h);
    return Py_BuildValue("O", flag ? Py_True : Py_False);
}


static PyObject* check_init(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    Py_BEGIN_ALLOW_THREADS
    oskar_imager_check_init(h, &status);
    Py_END_ALLOW_THREADS

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_check_init() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* coords_only(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int flag;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    flag = oskar_imager_coords_only(h);
    return Py_BuildValue("O", flag ? Py_True : Py_False);
}


static PyObject* create(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0, prec = 0;
    const char* type;
    if (!PyArg_ParseTuple(args, "s", &type)) return 0;
    prec = (type[0] == 'S' || type[0] == 's') ? OSKAR_SINGLE : OSKAR_DOUBLE;
    h = oskar_imager_create(prec, &status);
    capsule = PyCapsule_New((void*)h, name, (PyCapsule_Destructor)imager_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static oskar_Mem** create_cube(oskar_Imager* h, int plane_size, int plane_type,
        PyObject* dict, const char* key, int num_planes, int* status)
{
    oskar_Mem *alias_tmp, **cube_c;
    PyArrayObject *cube;
    int i, plane_elem;
    npy_intp dims[3];

    /* Create a Python array to hold the images. */
    plane_elem = plane_size * plane_size;
    dims[0]    = num_planes;
    dims[1]    = plane_size;
    dims[2]    = plane_size;
    cube       = (PyArrayObject*)PyArray_SimpleNew(3, dims,
            numpy_type_from_oskar(plane_type));
    if (!cube) return 0;

    /* Store the array in the dictionary. */
    PyDict_SetItemString(dict, key, (PyObject*)cube);

    /* Create the array of pointers to each plane for the imager. */
    alias_tmp = oskar_mem_create_alias_from_raw(PyArray_DATA(cube),
            plane_type, OSKAR_CPU, PyArray_SIZE(cube), status);
    cube_c = calloc(num_planes, sizeof(oskar_Mem*));
    for (i = 0; i < num_planes; ++i)
        cube_c[i] = oskar_mem_create_alias(alias_tmp,
                i * plane_elem, plane_elem, status);
    oskar_mem_free(alias_tmp, status);
    return cube_c;
}


static PyObject* fft_on_gpu(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("O", oskar_imager_fft_on_gpu(h) ? Py_True : Py_False);
}


static PyObject* finalise(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject *capsule = 0, *dict = 0;
    oskar_Mem **grids_c = 0, **images_c = 0;
    int i = 0, return_images = 0, return_grids = 0, status = 0;
    if (!PyArg_ParseTuple(args, "Oii",
            &capsule, &return_images, &return_grids))
        return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;

    /* Create a dictionary to return any outputs. */
    dict = PyDict_New();

    /* Check if we need to return images. */
    if (return_images > 0)
    {
        images_c = create_cube(h, oskar_imager_image_size(h),
                oskar_imager_precision(h), dict, "images", return_images,
                &status);
        if (!images_c) goto fail;
    }

    /* Check if we need to return grids. */
    if (return_grids > 0)
    {
        grids_c = create_cube(h, oskar_imager_plane_size(h),
                oskar_imager_plane_type(h), dict, "grids", return_grids,
                &status);
        if (!grids_c) goto fail;
    }

    /* Finalise. */
    Py_BEGIN_ALLOW_THREADS
    oskar_imager_finalise(h, return_images, images_c,
            return_grids, grids_c, &status);
    Py_END_ALLOW_THREADS

    /* Free handles. */
    if (grids_c)
    {
        for (i = 0; i < return_grids; ++i)
            oskar_mem_free(grids_c[i], &status);
        free(grids_c);
        grids_c = 0;
    }
    if (images_c)
    {
        for (i = 0; i < return_images; ++i)
            oskar_mem_free(images_c[i], &status);
        free(images_c);
        images_c = 0;
    }

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_finalise() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        goto fail;
    }
    return Py_BuildValue("N", dict); /* Don't increment refcount. */

fail:
    Py_XDECREF(dict);
    if (grids_c)
    {
        for (i = 0; i < return_grids; ++i)
            oskar_mem_free(grids_c[i], &status);
        free(grids_c);
        grids_c = 0;
    }
    if (images_c)
    {
        for (i = 0; i < return_images; ++i)
            oskar_mem_free(images_c[i], &status);
        free(images_c);
        images_c = 0;
    }
    return 0;
}


static PyObject* finalise_plane(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject *obj[] = {0, 0};
    PyArrayObject* plane = 0;
    oskar_Mem* plane_c = 0;
    int status = 0;
    double plane_norm = 0.0;
    if (!PyArg_ParseTuple(args, "OOd", &obj[0], &obj[1], &plane_norm))
        return 0;
    if (!(h = (oskar_Imager*) get_handle(obj[0], name))) return 0;

    /* Get the supplied plane. */
    plane = (PyArrayObject*) PyArray_FROM_OF(obj[1], NPY_ARRAY_OUT_ARRAY);
    if (!plane) goto fail;
    plane_c = oskar_mem_create_alias_from_raw(PyArray_DATA(plane),
            oskar_type_from_numpy(plane), OSKAR_CPU,
            PyArray_SIZE(plane), &status);

    /* Finalise the plane. */
    Py_BEGIN_ALLOW_THREADS
    oskar_imager_finalise_plane(h, plane_c, plane_norm, &status);
    Py_END_ALLOW_THREADS
    oskar_mem_free(plane_c, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_finalise_plane() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        goto fail;
    }
    Py_XDECREF(plane);
    return Py_BuildValue("");

fail:
    Py_XDECREF(plane);
    return 0;
}


static PyObject* fov(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_imager_fov(h));
}


static PyObject* freq_max_hz(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_imager_freq_max_hz(h));
}


static PyObject* freq_min_hz(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_imager_freq_min_hz(h));
}


static PyObject* generate_w_kernels_on_gpu(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("O",
            oskar_imager_generate_w_kernels_on_gpu(h) ? Py_True : Py_False);
}


static PyObject* image_size(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_imager_image_size(h));
}


static PyObject* image_type(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("s", oskar_imager_image_type(h));
}


static PyObject* input_file(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject *capsule = 0, *list = 0;
    int i, num_files = 0;
    char* const* files;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;

    /* Return the input file name or list of input file names. */
    num_files = oskar_imager_num_input_files(h);
    files = oskar_imager_input_files(h);
    if (num_files == 0) return Py_BuildValue("");
    if (num_files == 1) return Py_BuildValue("s", files[0]);
    list = PyList_New(num_files);
    for (i = 0; i < num_files; ++i)
    {
        PyList_SetItem(list, i, Py_BuildValue("s", files[i]));
    }
    return Py_BuildValue("N", list);
}


static PyObject* ms_column(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("s", oskar_imager_ms_column(h));
}


static PyObject* num_w_planes(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_imager_num_w_planes(h));
}


static PyObject* output_root(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("s", oskar_imager_output_root(h));
}


static PyObject* plane_size(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_imager_plane_size(h));
}


static PyObject* reset_cache(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_reset_cache(h, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_reset_cache() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* rotate_coords(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject *obj[] = {0, 0, 0, 0};
    oskar_Mem *uu_c, *vv_c, *ww_c;
    PyArrayObject *uu = 0, *vv = 0, *ww = 0;
    int status = 0;
    size_t num_coords;

    /* Parse inputs. */
    if (!PyArg_ParseTuple(args, "OOOO", &obj[0], &obj[1], &obj[2], &obj[3]))
        return 0;
    if (!(h = (oskar_Imager*) get_handle(obj[0], name))) return 0;

    /* Make sure input objects are arrays. Convert if required. */
    uu = (PyArrayObject*) PyArray_FROM_OF(obj[1], NPY_ARRAY_INOUT_ARRAY);
    vv = (PyArrayObject*) PyArray_FROM_OF(obj[2], NPY_ARRAY_INOUT_ARRAY);
    ww = (PyArrayObject*) PyArray_FROM_OF(obj[3], NPY_ARRAY_INOUT_ARRAY);
    if (!uu || !vv || !ww)
        goto fail;

    /* Check dimensions. */
    num_coords = (size_t) PyArray_SIZE(uu);
    if (num_coords != (size_t) PyArray_SIZE(vv) ||
            num_coords != (size_t) PyArray_SIZE(ww))
    {
        PyErr_SetString(PyExc_RuntimeError, "Data dimension mismatch.");
        goto fail;
    }

    /* Pointers to arrays. */
    uu_c = oskar_mem_create_alias_from_raw(PyArray_DATA(uu),
            oskar_type_from_numpy(uu), OSKAR_CPU, num_coords, &status);
    vv_c = oskar_mem_create_alias_from_raw(PyArray_DATA(vv),
            oskar_type_from_numpy(vv), OSKAR_CPU, num_coords, &status);
    ww_c = oskar_mem_create_alias_from_raw(PyArray_DATA(ww),
            oskar_type_from_numpy(ww), OSKAR_CPU, num_coords, &status);

    /* Rotate the baseline coordinates. */
    Py_BEGIN_ALLOW_THREADS
    oskar_imager_rotate_coords(h,
            num_coords, uu_c, vv_c, ww_c, uu_c, vv_c, ww_c);
    Py_END_ALLOW_THREADS
    oskar_mem_free(uu_c, &status);
    oskar_mem_free(vv_c, &status);
    oskar_mem_free(ww_c, &status);

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


static PyObject* rotate_vis(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject *obj[] = {0, 0, 0, 0, 0};
    oskar_Mem *uu_c, *vv_c, *ww_c, *amp_c;
    PyArrayObject *uu = 0, *vv = 0, *ww = 0, *amp = 0;
    int status = 0;
    size_t num_vis;

    /* Parse inputs. */
    if (!PyArg_ParseTuple(args, "OOOOO", &obj[0], &obj[1], &obj[2], &obj[3],
            &obj[4]))
        return 0;
    if (!(h = (oskar_Imager*) get_handle(obj[0], name))) return 0;

    /* Make sure input objects are arrays. Convert if required. */
    uu = (PyArrayObject*) PyArray_FROM_OF(obj[1], NPY_ARRAY_IN_ARRAY);
    vv = (PyArrayObject*) PyArray_FROM_OF(obj[2], NPY_ARRAY_IN_ARRAY);
    ww = (PyArrayObject*) PyArray_FROM_OF(obj[3], NPY_ARRAY_IN_ARRAY);
    amp = (PyArrayObject*) PyArray_FROM_OF(obj[4], NPY_ARRAY_INOUT_ARRAY);
    if (!uu || !vv || !ww || !amp)
        goto fail;

    /* Check dimensions. */
    num_vis = (size_t) PyArray_SIZE(amp);
    if (num_vis != (size_t) PyArray_SIZE(uu) ||
            num_vis != (size_t) PyArray_SIZE(vv) ||
            num_vis != (size_t) PyArray_SIZE(ww))
    {
        PyErr_SetString(PyExc_RuntimeError, "Data dimension mismatch.");
        goto fail;
    }

    /* Pointers to arrays. */
    uu_c = oskar_mem_create_alias_from_raw(PyArray_DATA(uu),
            oskar_type_from_numpy(uu), OSKAR_CPU, num_vis, &status);
    vv_c = oskar_mem_create_alias_from_raw(PyArray_DATA(vv),
            oskar_type_from_numpy(vv), OSKAR_CPU, num_vis, &status);
    ww_c = oskar_mem_create_alias_from_raw(PyArray_DATA(ww),
            oskar_type_from_numpy(ww), OSKAR_CPU, num_vis, &status);
    amp_c = oskar_mem_create_alias_from_raw(PyArray_DATA(amp),
            oskar_type_from_numpy(amp), OSKAR_CPU, num_vis, &status);

    /* Phase-rotate the visibility amplitudes. */
    Py_BEGIN_ALLOW_THREADS
    oskar_imager_rotate_vis(h, num_vis, uu_c, vv_c, ww_c, amp_c);
    Py_END_ALLOW_THREADS
    oskar_mem_free(uu_c, &status);
    oskar_mem_free(vv_c, &status);
    oskar_mem_free(ww_c, &status);
    oskar_mem_free(amp_c, &status);

    Py_XDECREF(uu);
    Py_XDECREF(vv);
    Py_XDECREF(ww);
    Py_XDECREF(amp);
    return Py_BuildValue("");

fail:
    Py_XDECREF(uu);
    Py_XDECREF(vv);
    Py_XDECREF(ww);
    Py_XDECREF(amp);
    return 0;
}


static PyObject* run(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject *capsule = 0, *dict = 0;
    oskar_Mem **grids_c = 0, **images_c = 0;
    int i = 0, return_images = 0, return_grids = 0, status = 0;
    if (!PyArg_ParseTuple(args, "Oii",
            &capsule, &return_images, &return_grids))
        return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;

    /* Create a dictionary to return any outputs. */
    dict = PyDict_New();

    /* Check if we need to return images. */
    if (return_images > 0)
    {
        images_c = create_cube(h, oskar_imager_image_size(h),
                oskar_imager_precision(h), dict, "images", return_images,
                &status);
        if (!images_c) goto fail;
    }

    /* Check if we need to return grids. */
    if (return_grids > 0)
    {
        grids_c = create_cube(h, oskar_imager_plane_size(h),
                oskar_imager_plane_type(h), dict, "grids", return_grids,
                &status);
        if (!grids_c) goto fail;
    }

    /* Run the imager. */
    Py_BEGIN_ALLOW_THREADS
    oskar_imager_run(h, return_images, images_c,
            return_grids, grids_c, &status);
    Py_END_ALLOW_THREADS

    /* Free handles. */
    if (grids_c)
    {
        for (i = 0; i < return_grids; ++i)
            oskar_mem_free(grids_c[i], &status);
        free(grids_c);
        grids_c = 0;
    }
    if (images_c)
    {
        for (i = 0; i < return_images; ++i)
            oskar_mem_free(images_c[i], &status);
        free(images_c);
        images_c = 0;
    }

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_run() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        goto fail;
    }
    return Py_BuildValue("N", dict); /* Don't increment refcount. */

fail:
    Py_XDECREF(dict);
    if (grids_c)
    {
        for (i = 0; i < return_grids; ++i)
            oskar_mem_free(grids_c[i], &status);
        free(grids_c);
        grids_c = 0;
    }
    if (images_c)
    {
        for (i = 0; i < return_images; ++i)
            oskar_mem_free(images_c[i], &status);
        free(images_c);
        images_c = 0;
    }
    return 0;
}


static PyObject* scale_norm_with_num_input_files(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("O",
            oskar_imager_scale_norm_with_num_input_files(h) ?
                    Py_True : Py_False);
}


static PyObject* set_algorithm(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* type = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &type)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_algorithm(h, type, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_set_algorithm() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* set_cellsize(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    double cellsize = 0.0;
    if (!PyArg_ParseTuple(args, "Od", &capsule, &cellsize)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_cellsize(h, cellsize);
    return Py_BuildValue("");
}


static PyObject* set_channel_snapshots(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_channel_snapshots(h, value);
    return Py_BuildValue("");
}


static PyObject* set_coords_only(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int flag = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &flag)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_coords_only(h, flag);
    return Py_BuildValue("");
}


static PyObject* set_default_direction(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_default_direction(h);
    return Py_BuildValue("");
}


static PyObject* set_direction(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    double ra = 0.0, dec = 0.0;
    if (!PyArg_ParseTuple(args, "Odd", &capsule, &ra, &dec)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_direction(h, ra, dec);
    return Py_BuildValue("");
}


static PyObject* set_fft_on_gpu(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_fft_on_gpu(h, value);
    return Py_BuildValue("");
}


static PyObject* set_fov(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    double fov = 0.0;
    if (!PyArg_ParseTuple(args, "Od", &capsule, &fov)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_fov(h, fov);
    return Py_BuildValue("");
}


static PyObject* set_freq_max_hz(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    double value = 0.0;
    if (!PyArg_ParseTuple(args, "Od", &capsule, &value)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_freq_max_hz(h, value);
    return Py_BuildValue("");
}


static PyObject* set_freq_min_hz(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    double value = 0.0;
    if (!PyArg_ParseTuple(args, "Od", &capsule, &value)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_freq_min_hz(h, value);
    return Py_BuildValue("");
}


static PyObject* set_generate_w_kernels_on_gpu(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_generate_w_kernels_on_gpu(h, value);
    return Py_BuildValue("");
}


static PyObject* set_grid_kernel(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0, support = 0, oversample = 0;
    const char* type = 0;
    if (!PyArg_ParseTuple(args, "Osii", &capsule, &type, &support, &oversample))
        return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_grid_kernel(h, type, support, oversample, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_set_grid_kernel() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* set_image_size(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int size = 0, status = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &size)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_image_size(h, size, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_set_image_size() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* set_image_type(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* type = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &type)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_image_type(h, type, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_set_image_type() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* set_input_file(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject *capsule = 0, *list = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &list)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;

    /* Check to see if the list is really a list, or a string. */
#if PY_MAJOR_VERSION >= 3
    if (PyUnicode_Check(list))
#else
    if (PyString_Check(list))
#endif
    {
#if PY_MAJOR_VERSION >= 3
        const char* file = PyUnicode_AsUTF8(list);
#else
        const char* file = PyString_AsString(list);
#endif
        oskar_imager_set_input_files(h, 1, &file, &status);
        return Py_BuildValue("i", status);
    }
    else if (PyList_Check(list))
    {
        int i, num_files;
        const char** files = 0;
        num_files = (int) PyList_Size(list);
        files = (const char**) calloc(num_files, sizeof(const char*));
        for (i = 0; i < num_files; ++i)
        {
#if PY_MAJOR_VERSION >= 3
            files[i] = PyUnicode_AsUTF8(PyList_GetItem(list, i));
#else
            files[i] = PyString_AsString(PyList_GetItem(list, i));
#endif
        }
        oskar_imager_set_input_files(h, num_files, files, &status);
        free(files);
        return Py_BuildValue("i", status);
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError,
                "Argument must be a string or list of strings.");
        return 0;
    }
}


static PyObject* set_ms_column(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* column = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &column)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_ms_column(h, column, &status);
    return Py_BuildValue("i", status);
}


static PyObject* set_num_w_planes(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int num = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &num)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_num_w_planes(h, num);
    return Py_BuildValue("");
}


static PyObject* set_output_root(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    const char* filename = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &filename)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_output_root(h, filename);
    return Py_BuildValue("");
}


static PyObject* set_scale_norm_with_num_input_files(PyObject* self,
        PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_scale_norm_with_num_input_files(h, value);
    return Py_BuildValue("");
}


static PyObject* set_size(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int size = 0, status = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &size)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_size(h, size, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_set_size() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* set_time_max_utc(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    double value = 0.0;
    if (!PyArg_ParseTuple(args, "Od", &capsule, &value)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_time_max_utc(h, value);
    return Py_BuildValue("");
}


static PyObject* set_time_min_utc(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    double value = 0.0;
    if (!PyArg_ParseTuple(args, "Od", &capsule, &value)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_time_min_utc(h, value);
    return Py_BuildValue("");
}


static PyObject* set_uv_filter_max(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    double value = 0.0;
    if (!PyArg_ParseTuple(args, "Od", &capsule, &value)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_uv_filter_max(h, value);
    return Py_BuildValue("");
}


static PyObject* set_uv_filter_min(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    double value = 0.0;
    if (!PyArg_ParseTuple(args, "Od", &capsule, &value)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_uv_filter_min(h, value);
    return Py_BuildValue("");
}


static PyObject* set_vis_frequency(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int num = 0;
    double ref = 0.0, inc = 0.0;
    if (!PyArg_ParseTuple(args, "Oddi", &capsule, &ref, &inc, &num)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_vis_frequency(h, ref, inc, num);
    return Py_BuildValue("");
}


static PyObject* set_vis_phase_centre(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    double ra = 0.0, dec = 0.0;
    if (!PyArg_ParseTuple(args, "Odd", &capsule, &ra, &dec)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_vis_phase_centre(h, ra, dec);
    return Py_BuildValue("");
}


static PyObject* set_weighting(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* type = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &type)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    oskar_imager_set_weighting(h, type, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_set_weighting() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* size(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_imager_size(h));
}


static PyObject* time_max_utc(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_imager_time_max_utc(h));
}


static PyObject* time_min_utc(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_imager_time_min_utc(h));
}


static PyObject* update(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject *obj[] = {0, 0, 0, 0, 0, 0, 0};
    oskar_Mem *uu_c, *vv_c, *ww_c, *amp_c = 0, *weight_c, *time_centroid_c = 0;
    PyArrayObject *uu = 0, *vv = 0, *ww = 0, *amps = 0;
    PyArrayObject *weight = 0, *time_centroid = 0;
    int start_chan = 0, end_chan = 0, num_pols = 1, status = 0;
    size_t num_rows, num_weights;

    /* Parse inputs. */
    if (!PyArg_ParseTuple(args, "OOOOOOOiii", &obj[0],
            &obj[1], &obj[2], &obj[3], &obj[4], &obj[5], &obj[6],
            &start_chan, &end_chan, &num_pols))
        return 0;
    if (!(h = (oskar_Imager*) get_handle(obj[0], name))) return 0;

    /* Make sure input objects are arrays. Convert if required. */
    uu     = (PyArrayObject*) PyArray_FROM_OF(obj[1], NPY_ARRAY_IN_ARRAY);
    vv     = (PyArrayObject*) PyArray_FROM_OF(obj[2], NPY_ARRAY_IN_ARRAY);
    ww     = (PyArrayObject*) PyArray_FROM_OF(obj[3], NPY_ARRAY_IN_ARRAY);
    if (!uu || !vv || !ww) goto fail;

    /* Check if visibility amplitudes are present. */
    if (obj[4] != Py_None)
    {
        amps = (PyArrayObject*) PyArray_FROM_OF(obj[4], NPY_ARRAY_IN_ARRAY);
        if (!amps) goto fail;

        /* Check visibility data are complex. */
        if (!PyArray_ISCOMPLEX(amps))
        {
            PyErr_SetString(PyExc_RuntimeError,
                    "Input visibility data must be complex.");
            goto fail;
        }
    }

    /* Check if weights are present. */
    if (obj[5] != Py_None)
    {
        weight = (PyArrayObject*) PyArray_FROM_OF(obj[5], NPY_ARRAY_IN_ARRAY);
        if (!weight) goto fail;
    }

    /* Check if time centroid values are present. */
    if (obj[6] != Py_None)
    {
        time_centroid = (PyArrayObject*) PyArray_FROM_OTF(obj[6],
                NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (!time_centroid) goto fail;
    }

    /* Check if visibilities are required but not present. */
    if (!oskar_imager_coords_only(h) && !amps)
    {
        PyErr_SetString(PyExc_RuntimeError, "Visibility data not present and "
                "imager not in coordinate-only mode.");
        goto fail;
    }

    /* Check number of polarisations. */
    if (num_pols != 1 && num_pols != 4)
    {
        PyErr_SetString(PyExc_ValueError,
                "Unknown number of polarisations. Must be 1 or 4.");
        goto fail;
    }

    /* Get dimensions. */
    num_rows = (size_t) PyArray_SIZE(uu);
    if (num_rows != (size_t) PyArray_SIZE(vv) ||
            num_rows != (size_t) PyArray_SIZE(ww))
    {
        PyErr_SetString(PyExc_RuntimeError,
                "Coordinate data dimension mismatch.");
        goto fail;
    }
    num_weights = num_rows * num_pols;

    /* Pointers to input arrays. */
    uu_c = oskar_mem_create_alias_from_raw(PyArray_DATA(uu),
            oskar_type_from_numpy(uu), OSKAR_CPU, num_rows, &status);
    vv_c = oskar_mem_create_alias_from_raw(PyArray_DATA(vv),
            oskar_type_from_numpy(vv), OSKAR_CPU, num_rows, &status);
    ww_c = oskar_mem_create_alias_from_raw(PyArray_DATA(ww),
            oskar_type_from_numpy(ww), OSKAR_CPU, num_rows, &status);
    if (amps)
    {
        int vis_type;
        size_t num_vis;
        vis_type = oskar_type_from_numpy(amps);
        num_vis = num_rows * (1 + end_chan - start_chan);
        if (num_pols == 4) vis_type |= OSKAR_MATRIX;
        amp_c = oskar_mem_create_alias_from_raw(PyArray_DATA(amps),
                vis_type, OSKAR_CPU, num_vis, &status);
    }
    if (weight)
    {
        weight_c = oskar_mem_create_alias_from_raw(PyArray_DATA(weight),
                oskar_type_from_numpy(weight), OSKAR_CPU, num_weights, &status);
    }
    else
    {
        /* Set weights to 1 if not supplied. */
        weight_c = oskar_mem_create(oskar_imager_precision(h), OSKAR_CPU,
                num_weights, &status);
        oskar_mem_set_value_real(weight_c, 1.0, 0, num_weights, &status);
    }
    if (time_centroid)
    {
        time_centroid_c = oskar_mem_create_alias_from_raw(
                PyArray_DATA(time_centroid),
                oskar_type_from_numpy(time_centroid),
                OSKAR_CPU, num_rows, &status);
    }

    /* Update the imager with the supplied visibility data. */
    Py_BEGIN_ALLOW_THREADS
    oskar_imager_update(h, num_rows, start_chan, end_chan, num_pols,
            uu_c, vv_c, ww_c, amp_c, weight_c, time_centroid_c, &status);
    Py_END_ALLOW_THREADS
    oskar_mem_free(uu_c, &status);
    oskar_mem_free(vv_c, &status);
    oskar_mem_free(ww_c, &status);
    oskar_mem_free(amp_c, &status);
    oskar_mem_free(weight_c, &status);
    oskar_mem_free(time_centroid_c, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_update() failed with code %d (%s).", status,
                oskar_get_error_string(status));
        goto fail;
    }
    Py_XDECREF(uu);
    Py_XDECREF(vv);
    Py_XDECREF(ww);
    Py_XDECREF(amps);
    Py_XDECREF(weight);
    Py_XDECREF(time_centroid);
    return Py_BuildValue("");

fail:
    Py_XDECREF(uu);
    Py_XDECREF(vv);
    Py_XDECREF(ww);
    Py_XDECREF(amps);
    Py_XDECREF(weight);
    Py_XDECREF(time_centroid);
    return 0;
}


static PyObject* update_from_block(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    oskar_VisBlock* block = 0;
    oskar_VisHeader* header = 0;
    PyObject *obj[] = {0, 0, 0};
    int status = 0;

    /* Parse inputs. */
    if (!PyArg_ParseTuple(args, "OOO", &obj[0], &obj[1], &obj[2]))
        return 0;
    if (!(h = (oskar_Imager*) get_handle(obj[0], name))) return 0;
    if (!(header = (oskar_VisHeader*) get_handle(obj[1], "oskar_VisHeader")))
        return 0;
    if (!(block = (oskar_VisBlock*) get_handle(obj[2], "oskar_VisBlock")))
        return 0;

    /* Update the imager with the supplied visibility data. */
    Py_BEGIN_ALLOW_THREADS
    oskar_imager_update_from_block(h, header, block, &status);
    Py_END_ALLOW_THREADS

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_update_from_block() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* update_plane(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject *obj[] = {0, 0, 0, 0, 0, 0, 0, 0};
    oskar_Mem *uu_c = 0, *vv_c = 0, *ww_c = 0, *amp_c = 0, *weight_c = 0;
    oskar_Mem *plane_c = 0, *weights_grid_c = 0;
    PyArrayObject *uu = 0, *vv = 0, *ww = 0, *amps = 0, *weight = 0;
    PyArrayObject *plane = 0, *weights_grid = 0;
    double plane_norm = 0.0;
    int status = 0;
    size_t num_vis;

    /* Parse inputs:
     * capsule, uu, vv, ww, amps, weight, plane, plane_norm, weights_grid. */
    if (!PyArg_ParseTuple(args, "OOOOOOOdO", &obj[0],
            &obj[1], &obj[2], &obj[3], &obj[4], &obj[5], &obj[6], &plane_norm,
            &obj[7]))
        return 0;
    if (!(h = (oskar_Imager*) get_handle(obj[0], name))) return 0;

    /* Make sure required input objects are arrays. Convert if required. */
    uu     = (PyArrayObject*) PyArray_FROM_OF(obj[1], NPY_ARRAY_IN_ARRAY);
    vv     = (PyArrayObject*) PyArray_FROM_OF(obj[2], NPY_ARRAY_IN_ARRAY);
    ww     = (PyArrayObject*) PyArray_FROM_OF(obj[3], NPY_ARRAY_IN_ARRAY);
    if (!uu || !vv || !ww) goto fail;

    /* Check if visibility amplitudes are present. */
    if (obj[4] != Py_None)
    {
        amps = (PyArrayObject*) PyArray_FROM_OF(obj[4], NPY_ARRAY_IN_ARRAY);
        if (!amps) goto fail;

        /* Check visibility data are complex. */
        if (!PyArray_ISCOMPLEX(amps))
        {
            PyErr_SetString(PyExc_RuntimeError,
                    "Input visibility data must be complex.");
            goto fail;
        }
    }

    /* Check if weights are present. */
    if (obj[5] != Py_None)
    {
        weight = (PyArrayObject*) PyArray_FROM_OF(obj[5], NPY_ARRAY_IN_ARRAY);
        if (!weight) goto fail;
    }

    /* Check if visibility grid is present. */
    if (obj[6] != Py_None)
    {
        plane = (PyArrayObject*) PyArray_FROM_OF(obj[6], NPY_ARRAY_IN_ARRAY);
        if (!plane) goto fail;
    }

    /* Check if weights grid is present. */
    if (obj[7] != Py_None)
    {
        weights_grid = (PyArrayObject*) PyArray_FROM_OF(obj[7],
                NPY_ARRAY_IN_ARRAY);
        if (!weights_grid) goto fail;
    }

    /* Check if visibilities are required but not present. */
    if (!oskar_imager_coords_only(h) && (!amps || !plane))
    {
        PyErr_SetString(PyExc_RuntimeError, "Visibility data not present and "
                "imager not in coordinate-only mode.");
        goto fail;
    }

    /* Check dimensions. */
    num_vis = (size_t) PyArray_SIZE(uu);
    if (num_vis != (size_t) PyArray_SIZE(vv) ||
            num_vis != (size_t) PyArray_SIZE(ww) ||
            (amps && num_vis != (size_t) PyArray_SIZE(amps)) ||
            (weight && num_vis != (size_t) PyArray_SIZE(weight)))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input data dimension mismatch.");
        goto fail;
    }

    /* Pointers to input arrays. */
    uu_c = oskar_mem_create_alias_from_raw(PyArray_DATA(uu),
            oskar_type_from_numpy(uu), OSKAR_CPU, num_vis, &status);
    vv_c = oskar_mem_create_alias_from_raw(PyArray_DATA(vv),
            oskar_type_from_numpy(vv), OSKAR_CPU, num_vis, &status);
    ww_c = oskar_mem_create_alias_from_raw(PyArray_DATA(ww),
            oskar_type_from_numpy(ww), OSKAR_CPU, num_vis, &status);
    if (amps)
    {
        amp_c = oskar_mem_create_alias_from_raw(PyArray_DATA(amps),
                oskar_type_from_numpy(amps), OSKAR_CPU, num_vis, &status);
    }
    if (weight)
    {
        weight_c = oskar_mem_create_alias_from_raw(PyArray_DATA(weight),
                oskar_type_from_numpy(weight), OSKAR_CPU, num_vis, &status);
    }
    else
    {
        /* Set weights to 1 if not supplied. */
        weight_c = oskar_mem_create(oskar_imager_precision(h), OSKAR_CPU,
                num_vis, &status);
        oskar_mem_set_value_real(weight_c, 1.0, 0, num_vis, &status);
    }
    if (plane)
    {
        plane_c = oskar_mem_create_alias_from_raw(PyArray_DATA(plane),
                oskar_type_from_numpy(plane), OSKAR_CPU,
                (size_t) PyArray_SIZE(plane), &status);
    }
    if (weights_grid)
    {
        weights_grid_c = oskar_mem_create_alias_from_raw(
                PyArray_DATA(weights_grid),
                oskar_type_from_numpy(weights_grid), OSKAR_CPU,
                (size_t) PyArray_SIZE(weights_grid), &status);
    }

    /* Update the plane. */
    Py_BEGIN_ALLOW_THREADS
    oskar_imager_update_plane(h, num_vis, uu_c, vv_c, ww_c, amp_c,
            weight_c, plane_c, &plane_norm, weights_grid_c, &status);
    Py_END_ALLOW_THREADS

    /* Clean up. */
    oskar_mem_free(uu_c, &status);
    oskar_mem_free(vv_c, &status);
    oskar_mem_free(ww_c, &status);
    oskar_mem_free(amp_c, &status);
    oskar_mem_free(weight_c, &status);
    oskar_mem_free(plane_c, &status);
    oskar_mem_free(weights_grid_c, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_imager_update_plane() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        goto fail;
    }
    Py_XDECREF(uu);
    Py_XDECREF(vv);
    Py_XDECREF(ww);
    Py_XDECREF(amps);
    Py_XDECREF(weight);
    Py_XDECREF(plane);
    Py_XDECREF(weights_grid);
    return Py_BuildValue("d", plane_norm);

fail:
    Py_XDECREF(uu);
    Py_XDECREF(vv);
    Py_XDECREF(ww);
    Py_XDECREF(amps);
    Py_XDECREF(weight);
    Py_XDECREF(plane);
    Py_XDECREF(weights_grid);
    return 0;
}


static PyObject* uv_filter_max(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_imager_uv_filter_max(h));
}


static PyObject* uv_filter_min(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("d", oskar_imager_uv_filter_min(h));
}


static PyObject* weighting(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Imager*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("s", oskar_imager_weighting(h));
}


static PyObject* make_image(PyObject* self, PyObject* args)
{
    oskar_Imager* h;
    PyObject *obj[] = {0, 0, 0, 0, 0};
    PyArrayObject *uu = 0, *vv = 0, *ww = 0, *amps = 0, *weight = 0, *im = 0;
    size_t num_cells, num_pixels, num_vis;
    int plane_size = 0, size = 0;
    int dft = 0, status = 0, type = 0, wproj = 0, uniform = 0, wprojplanes = -1;
    double fov_deg = 0.0, norm = 0.0;
    const char *weighting_type = 0, *algorithm_type = 0;
    oskar_Mem *uu_c, *vv_c, *ww_c, *amp_c, *weight_c, *plane;
    oskar_Mem *weights_grid = 0;
    npy_intp dims[2];

    /* Parse inputs. */
    if (!PyArg_ParseTuple(args, "OOOOdissOi",
            &obj[0], &obj[1], &obj[2], &obj[3], &fov_deg, &size,
            &weighting_type, &algorithm_type, &obj[4], &wprojplanes))
        return 0;

    /* Make sure input objects are arrays. Convert if required. */
    uu     = (PyArrayObject*) PyArray_FROM_OF(obj[0], NPY_ARRAY_IN_ARRAY);
    vv     = (PyArrayObject*) PyArray_FROM_OF(obj[1], NPY_ARRAY_IN_ARRAY);
    ww     = (PyArrayObject*) PyArray_FROM_OF(obj[2], NPY_ARRAY_IN_ARRAY);
    amps   = (PyArrayObject*) PyArray_FROM_OF(obj[3], NPY_ARRAY_IN_ARRAY);
    if (!uu || !vv || !ww || !amps) goto fail;

    /* Check if weights are present. */
    if (obj[4] != Py_None)
    {
        weight = (PyArrayObject*) PyArray_FROM_OF(obj[4], NPY_ARRAY_IN_ARRAY);
        if (!weight) goto fail;
    }

    /* Check dimensions. */
    num_pixels = size * size;
    num_vis = (size_t) PyArray_SIZE(amps);
    if (num_vis != (size_t) PyArray_SIZE(uu) ||
            num_vis != (size_t) PyArray_SIZE(vv) ||
            num_vis != (size_t) PyArray_SIZE(ww) ||
            (weight && num_vis != (size_t) PyArray_SIZE(weight)))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input data dimension mismatch.");
        goto fail;
    }

    /* Get precision of complex visibility data. */
    if (!PyArray_ISCOMPLEX(amps))
    {
        PyErr_SetString(PyExc_RuntimeError,
                "Input visibility data must be complex.");
        goto fail;
    }
    type = oskar_type_precision(oskar_type_from_numpy(amps));

    /* Pointers to input/output arrays. */
    uu_c = oskar_mem_create_alias_from_raw(PyArray_DATA(uu),
            oskar_type_from_numpy(uu), OSKAR_CPU, num_vis, &status);
    vv_c = oskar_mem_create_alias_from_raw(PyArray_DATA(vv),
            oskar_type_from_numpy(vv), OSKAR_CPU, num_vis, &status);
    ww_c = oskar_mem_create_alias_from_raw(PyArray_DATA(ww),
            oskar_type_from_numpy(ww), OSKAR_CPU, num_vis, &status);
    amp_c = oskar_mem_create_alias_from_raw(PyArray_DATA(amps),
            oskar_type_from_numpy(amps), OSKAR_CPU, num_vis, &status);
    if (weight)
    {
        weight_c = oskar_mem_create_alias_from_raw(PyArray_DATA(weight),
                oskar_type_from_numpy(weight), OSKAR_CPU, num_vis, &status);
    }
    else
    {
        /* Set weights to 1 if not supplied. */
        weight_c = oskar_mem_create(type, OSKAR_CPU, num_vis, &status);
        oskar_mem_set_value_real(weight_c, 1.0, 0, num_vis, &status);
    }

    /* Allow threads. */
    Py_BEGIN_ALLOW_THREADS

    /* Create and set up the imager. */
    h = oskar_imager_create(type, &status);
    oskar_imager_set_fov(h, fov_deg);
    oskar_imager_set_size(h, size, &status);
    oskar_imager_set_algorithm(h, algorithm_type, &status);
    oskar_imager_set_num_w_planes(h, wprojplanes);
    oskar_imager_set_weighting(h, weighting_type, &status);

    /* Check for DFT, W-projection or uniform weighting. */
    if (!strncmp(algorithm_type, "DFT", 3) ||
            !strncmp(algorithm_type, "dft", 3))
        dft = 1;
    if (!strncmp(algorithm_type, "W", 1) ||
            !strncmp(algorithm_type, "w", 1))
        wproj = 1;
    if (!strncmp(weighting_type, "U", 1) ||
            !strncmp(weighting_type, "u", 1))
        uniform = 1;

    /* Get the plane size. */
    plane_size = oskar_imager_plane_size(h);
    num_cells = plane_size * plane_size;

    /* Supply the coordinates first, if required. */
    if (wproj || uniform)
    {
        weights_grid = oskar_mem_create(type, OSKAR_CPU, num_cells, &status);
        oskar_imager_set_coords_only(h, 1);
        oskar_imager_update_plane(h, num_vis, uu_c, vv_c, ww_c, 0, weight_c,
                0, 0, weights_grid, &status);
        oskar_imager_set_coords_only(h, 0);
    }

    /* Initialise the algorithm. */
    oskar_imager_check_init(h, &status);

    /* Make the image. */
    plane = oskar_mem_create((dft ? type : (type | OSKAR_COMPLEX)), OSKAR_CPU,
            num_cells, &status);
    oskar_imager_update_plane(h, num_vis, uu_c, vv_c, ww_c, amp_c, weight_c,
            plane, &norm, weights_grid, &status);
    oskar_imager_finalise_plane(h, plane, norm, &status);
    oskar_imager_trim_image(h, plane, plane_size, size, &status);

    /* Free temporaries. */
    oskar_mem_free(uu_c, &status);
    oskar_mem_free(vv_c, &status);
    oskar_mem_free(ww_c, &status);
    oskar_mem_free(amp_c, &status);
    oskar_mem_free(weight_c, &status);
    oskar_mem_free(weights_grid, &status);
    oskar_imager_free(h, &status);

    /* Disallow threads. */
    Py_END_ALLOW_THREADS

    /* Copy the data out. */
    dims[0] = size;
    dims[1] = size;
    im = (PyArrayObject*)PyArray_SimpleNew(2, dims,
            oskar_type_is_double(type) ? NPY_DOUBLE : NPY_FLOAT);
    memcpy(PyArray_DATA(im), oskar_mem_void_const(plane),
            num_pixels * oskar_mem_element_size(type));
    oskar_mem_free(plane, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "make_image() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        goto fail;
    }

    /* Return image to the python workspace. */
    Py_XDECREF(uu);
    Py_XDECREF(vv);
    Py_XDECREF(ww);
    Py_XDECREF(amps);
    Py_XDECREF(weight);
    return Py_BuildValue("N", im); /* Don't increment refcount. */

fail:
    Py_XDECREF(uu);
    Py_XDECREF(vv);
    Py_XDECREF(ww);
    Py_XDECREF(amps);
    Py_XDECREF(weight);
    Py_XDECREF(im);
    return 0;
}


/* Method table. */
static PyMethodDef methods[] =
{
        {"algorithm", (PyCFunction)algorithm, METH_VARARGS, "algorithm()"},
        {"capsule_name", (PyCFunction)capsule_name,
                METH_VARARGS, "capsule_name()"},
        {"cellsize", (PyCFunction)cellsize, METH_VARARGS, "cellsize()"},
        {"channel_snapshots", (PyCFunction)channel_snapshots,
                METH_VARARGS, "channel_snapshots()"},
        {"check_init", (PyCFunction)check_init, METH_VARARGS, "check_init()"},
        {"coords_only", (PyCFunction)coords_only,
                METH_VARARGS, "coords_only()"},
        {"create", (PyCFunction)create, METH_VARARGS, "create(type)"},
        {"fft_on_gpu", (PyCFunction)fft_on_gpu, METH_VARARGS, "fft_on_gpu()"},
        {"finalise", (PyCFunction)finalise,
                METH_VARARGS, "finalise(return_images, return_grids)"},
        {"finalise_plane", (PyCFunction)finalise_plane,
                METH_VARARGS, "finalise_plane(plane, plane_norm)"},
        {"fov", (PyCFunction)fov, METH_VARARGS, "fov()"},
        {"freq_max_hz", (PyCFunction)freq_max_hz,
                METH_VARARGS, "freq_max_hz()"},
        {"freq_min_hz", (PyCFunction)freq_min_hz,
                METH_VARARGS, "freq_min_hz()"},
        {"generate_w_kernels_on_gpu", (PyCFunction)generate_w_kernels_on_gpu,
                METH_VARARGS, "generate_w_kernels_on_gpu()"},
        {"image_size", (PyCFunction)image_size, METH_VARARGS, "image_size()"},
        {"image_type", (PyCFunction)image_type, METH_VARARGS, "image_type()"},
        {"input_file", (PyCFunction)input_file, METH_VARARGS, "input_file()"},
        {"ms_column", (PyCFunction)ms_column, METH_VARARGS, "ms_column()"},
        {"make_image", (PyCFunction)make_image, METH_VARARGS,
                "make_image(uu, vv, ww, amp, weight, fov_deg, size)"},
        {"num_w_planes", (PyCFunction)num_w_planes,
                METH_VARARGS, "num_w_planes()"},
        {"output_root", (PyCFunction)output_root,
                METH_VARARGS, "output_root()"},
        {"plane_size", (PyCFunction)plane_size, METH_VARARGS, "plane_size()"},
        {"reset_cache", (PyCFunction)reset_cache,
                METH_VARARGS, "reset_cache()"},
        {"rotate_coords", (PyCFunction)rotate_coords,
                METH_VARARGS, "rotate_coords(uu, vv, ww)"},
        {"rotate_vis", (PyCFunction)rotate_vis,
                METH_VARARGS, "rotate_vis(uu_in, vv_in, ww_in, amps)"},
        {"run", (PyCFunction)run,
                METH_VARARGS, "run(return_images, return_grids)"},
        {"scale_norm_with_num_input_files",
                (PyCFunction)scale_norm_with_num_input_files,
                METH_VARARGS, "scale_norm_with_num_input_files()"},
        {"set_algorithm", (PyCFunction)set_algorithm,
                METH_VARARGS, "set_algorithm(type)"},
        {"set_cellsize", (PyCFunction)set_cellsize,
                METH_VARARGS, "set_cellsize(value)"},
        {"set_channel_snapshots", (PyCFunction)set_channel_snapshots,
                METH_VARARGS, "set_channel_snapshots(value)"},
        {"set_coords_only", (PyCFunction)set_coords_only,
                METH_VARARGS, "set_coords_only(flag)"},
        {"set_default_direction", (PyCFunction)set_default_direction,
                METH_VARARGS, "set_default_direction()"},
        {"set_direction", (PyCFunction)set_direction,
                METH_VARARGS, "set_direction(ra_deg, dec_deg)"},
        {"set_fft_on_gpu", (PyCFunction)set_fft_on_gpu,
                METH_VARARGS, "set_fft_on_gpu(value)"},
        {"set_fov", (PyCFunction)set_fov, METH_VARARGS, "set_fov(value)"},
        {"set_freq_max_hz", (PyCFunction)set_freq_max_hz,
                METH_VARARGS, "set_freq_max_hz(value)"},
        {"set_freq_min_hz", (PyCFunction)set_freq_min_hz,
                METH_VARARGS, "set_freq_min_hz(value)"},
        {"set_generate_w_kernels_on_gpu",
                (PyCFunction)set_generate_w_kernels_on_gpu,
                METH_VARARGS, "set_generate_w_kernels_on_gpu(value)"},
        {"set_grid_kernel", (PyCFunction)set_grid_kernel,
                METH_VARARGS, "set_grid_kernel(type, support, oversample)"},
        {"set_image_size", (PyCFunction)set_image_size,
                METH_VARARGS, "set_image_size(value)"},
        {"set_image_type", (PyCFunction)set_image_type,
                METH_VARARGS, "set_image_type(type)"},
        {"set_input_file", (PyCFunction)set_input_file,
                METH_VARARGS, "set_input_file(filename)"},
        {"set_ms_column", (PyCFunction)set_ms_column,
                METH_VARARGS, "set_ms_column(column)"},
        {"set_num_w_planes", (PyCFunction)set_num_w_planes,
                METH_VARARGS, "set_num_w_planes(value)"},
        {"set_output_root", (PyCFunction)set_output_root,
                METH_VARARGS, "set_output_root(filename)"},
        {"set_scale_norm_with_num_input_files",
                (PyCFunction)set_scale_norm_with_num_input_files,
                METH_VARARGS, "set_scale_norm_with_num_input_files(value)"},
        {"set_size", (PyCFunction)set_size, METH_VARARGS, "set_size(value)"},
        {"set_time_max_utc", (PyCFunction)set_time_max_utc,
                METH_VARARGS, "set_time_max_utc(value)"},
        {"set_time_min_utc", (PyCFunction)set_time_min_utc,
                METH_VARARGS, "set_time_min_utc(value)"},
        {"set_uv_filter_max", (PyCFunction)set_uv_filter_max,
                METH_VARARGS, "set_uv_filter_max(max_wavelengths)"},
        {"set_uv_filter_min", (PyCFunction)set_uv_filter_min,
                METH_VARARGS, "set_uv_filter_min(min_wavelengths)"},
        {"set_vis_frequency", (PyCFunction)set_vis_frequency, METH_VARARGS,
                "set_vis_frequency(ref_hz, inc_hz, num_channels)"},
        {"set_vis_phase_centre", (PyCFunction)set_vis_phase_centre,
                METH_VARARGS, "set_vis_phase_centre(ra_deg, dec_deg)"},
        {"set_weighting", (PyCFunction)set_weighting,
                METH_VARARGS, "set_weighting(type)"},
        {"size", (PyCFunction)size, METH_VARARGS, "size()"},
        {"time_max_utc", (PyCFunction)time_max_utc,
                METH_VARARGS, "time_max_utc()"},
        {"time_min_utc", (PyCFunction)time_min_utc,
                METH_VARARGS, "time_min_utc()"},
        {"update", (PyCFunction)update, METH_VARARGS,
                "update(uu, vv, ww, amps, weight, time_centroid, "
                "start_chan, end_chan, num_pols)"},
        {"update_from_block", (PyCFunction)update_from_block,
                METH_VARARGS, "update_from_block(vis_header, vis_block)"},
        {"update_plane", (PyCFunction)update_plane, METH_VARARGS,
                "update_plane(uu, vv, ww, amps, weight, plane, plane_norm)"},
        {"uv_filter_max", (PyCFunction)uv_filter_max,
                METH_VARARGS, "uv_filter_max()"},
        {"uv_filter_min", (PyCFunction)uv_filter_min,
                METH_VARARGS, "uv_filter_min()"},
        {"weighting", (PyCFunction)weighting, METH_VARARGS, "weighting()"},
        {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_imager_lib",      /* m_name */
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
    m = Py_InitModule3("_imager_lib", methods, module_doc);
#endif
    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__imager_lib(void)
{
    import_array();
    return moduleinit();
}
#else
/* The init function name has to match that of the compiled module
 * with the pattern 'init<module name>'. This module is called '_imager_lib' */
PyMODINIT_FUNC init_imager_lib(void)
{
    import_array();
    moduleinit();
    return;
}
#endif

