/*
 * Copyright (c) 2014-2016, The University of Oxford
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

#include <oskar_imager.h>
#include <oskar_get_error_string.h>
#include <string.h>

/* http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static const char* name = "oskar_Imager";

static void imager_free(PyObject* capsule)
{
    int status = 0;
    oskar_Imager* h = (oskar_Imager*) PyCapsule_GetPointer(capsule, name);
    oskar_imager_free(h, &status);
}


static oskar_Imager* get_handle(PyObject* capsule)
{
    oskar_Imager* h = 0;
    if (!PyCapsule_CheckExact(capsule))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input is not a PyCapsule object!");
        return 0;
    }
    h = (oskar_Imager*) PyCapsule_GetPointer(capsule, name);
    if (!h)
    {
        PyErr_SetString(PyExc_RuntimeError,
                "Unable to convert PyCapsule object to pointer.");
        return 0;
    }
    return h;
}


static PyObject* create(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0, prec = 0;
    const char* type;
    if (!PyArg_ParseTuple(args, "s", &type)) return 0;
    prec = (type[0] == 'D' || type[0] == 'd') ? OSKAR_DOUBLE : OSKAR_SINGLE;
    h = oskar_imager_create(prec, &status);
    capsule = PyCapsule_New((void*)h, name, (PyCapsule_Destructor)imager_free);
    return Py_BuildValue("Ni", capsule, status); /* Don't increment refcount. */
}


static PyObject* finalise(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    oskar_Mem* p = 0;
    PyObject* capsule = 0, *plane = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O|O", &capsule, &plane)) return 0;
    if (!(h = get_handle(capsule))) return 0;

    if (plane && plane != Py_None)
    {
        p = oskar_mem_create_alias_from_raw(
                PyArray_DATA((PyArrayObject*)plane), OSKAR_DOUBLE,
                OSKAR_CPU, 0, &status);
    }
    oskar_imager_finalise(h, p, &status);
    oskar_mem_free(p, &status);
    return Py_BuildValue("i", status);
}


static PyObject* reset_cache(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_reset_cache(h, &status);
    return Py_BuildValue("i", status);
}


static PyObject* run(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* filename = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &filename)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_run(h, filename, &status);
    return Py_BuildValue("i", status);
}


static PyObject* set_algorithm(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* type = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &type)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_algorithm(h, type, &status);
    return Py_BuildValue("i", status);
}


static PyObject* set_channel_range(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int start = 0, end = 0, snapshots = 0;
    if (!PyArg_ParseTuple(args, "Oiii", &capsule, &start, &end, &snapshots))
        return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_channel_range(h, start, end, snapshots);
    return Py_BuildValue("");
}


static PyObject* set_default_direction(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_default_direction(h);
    return Py_BuildValue("");
}


static PyObject* set_direction(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    double ra = 0.0, dec = 0.0;
    if (!PyArg_ParseTuple(args, "Odd", &capsule, &ra, &dec)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_direction(h, ra, dec);
    return Py_BuildValue("");
}


static PyObject* set_fov(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    double fov = 0.0;
    if (!PyArg_ParseTuple(args, "Od", &capsule, &fov)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_fov(h, fov);
    return Py_BuildValue("");
}


static PyObject* set_fft_on_gpu(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_fft_on_gpu(h, value);
    return Py_BuildValue("");
}


static PyObject* set_image_type(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* type = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &type)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_image_type(h, type, &status);
    return Py_BuildValue("i", status);
}


static PyObject* set_grid_kernel(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0, support = 0, oversample = 0;
    const char* type = 0;
    if (!PyArg_ParseTuple(args, "Osii", &capsule, &type, &support, &oversample))
        return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_grid_kernel(h, type, support, oversample, &status);
    return Py_BuildValue("i", status);
}


static PyObject* set_ms_column(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* column = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &column)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_ms_column(h, column, &status);
    return Py_BuildValue("i", status);
}


static PyObject* set_output_root(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* filename = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &filename)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_output_root(h, filename, &status);
    return Py_BuildValue("i", status);
}


static PyObject* set_size(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int size = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &size)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_size(h, size);
    return Py_BuildValue("");
}


static PyObject* set_time_range(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int start = 0, end = 0, snapshots = 0;
    if (!PyArg_ParseTuple(args, "Oiii", &capsule, &start, &end, &snapshots))
        return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_time_range(h, start, end, snapshots);
    return Py_BuildValue("");
}


static PyObject* set_vis_frequency(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0, num = 0;
    double ref = 0.0, inc = 0.0;
    if (!PyArg_ParseTuple(args, "Oddi", &capsule, &ref, &inc, &num)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_vis_frequency(h, ref, inc, num, &status);
    return Py_BuildValue("i", status);
}


static PyObject* set_vis_phase_centre(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    double ra = 0.0, dec = 0.0;
    if (!PyArg_ParseTuple(args, "Odd", &capsule, &ra, &dec)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_vis_phase_centre(h, ra, dec);
    return Py_BuildValue("");
}


static PyObject* set_vis_time(PyObject* self, PyObject* args)
{
    oskar_Imager* h = 0;
    PyObject* capsule = 0;
    int status = 0, num = 0;
    double ref = 0.0, inc = 0.0;
    if (!PyArg_ParseTuple(args, "Oddi", &capsule, &ref, &inc, &num)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    oskar_imager_set_vis_time(h, ref, inc, num, &status);
    return Py_BuildValue("i", status);
}


static PyObject* update(PyObject* self, PyObject* args, PyObject* keywds)
{
    oskar_Imager* h = 0;
    oskar_Mem *uu_c, *vv_c, *ww_c, *amp_c, *weight_c;
    PyObject* capsule = 0;
    PyArrayObject *uu = 0, *vv = 0, *ww = 0, *amps = 0, *weight = 0;
    int start_time = 0, end_time = 0, start_chan = 0, end_chan = 0;
    int num_pols = 1, num_baselines = 0, vis_type, type = OSKAR_DOUBLE;
    int num_times, num_chan, num_coords, num_vis, status = 0;

    /* Parse inputs. */
    char* keywords[] = {"_capsule", "num_baselines", "uu", "vv", "ww", "amps",
            "weight", "num_pols", "start_time", "end_time",
            "start_chan", "end_chan", 0};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OiOOOOO|iiiii", keywords,
            &capsule, &num_baselines, &uu, &vv, &ww, &amps, &weight,
            &num_pols, &start_time, &end_time, &start_chan, &end_chan))
        return 0;
    if (!(h = get_handle(capsule))) return 0;

    /* Get dimensions. */
    num_times = 1 + end_time - start_time;
    num_chan = 1 + end_chan - start_chan;
    num_coords = num_times * num_baselines;
    num_vis = num_coords * num_chan;
    vis_type = type | OSKAR_COMPLEX;
    if (num_pols == 4) vis_type |= OSKAR_MATRIX;

    /* Pointers to input arrays. */
    uu_c = oskar_mem_create_alias_from_raw(PyArray_DATA(uu), type,
            OSKAR_CPU, num_coords, &status);
    vv_c = oskar_mem_create_alias_from_raw(PyArray_DATA(vv), type,
            OSKAR_CPU, num_coords, &status);
    ww_c = oskar_mem_create_alias_from_raw(PyArray_DATA(ww), type,
            OSKAR_CPU, num_coords, &status);
    amp_c = oskar_mem_create_alias_from_raw(PyArray_DATA(amps), vis_type,
            OSKAR_CPU, num_vis, &status);
    weight_c = oskar_mem_create_alias_from_raw(PyArray_DATA(weight), type,
            OSKAR_CPU, num_vis, &status);

    oskar_imager_update(h, start_time, end_time, start_chan, end_chan,
            num_pols, num_baselines, uu_c, vv_c, ww_c, amp_c, weight_c,
            &status);
    oskar_mem_free(uu_c, &status);
    oskar_mem_free(vv_c, &status);
    oskar_mem_free(ww_c, &status);
    oskar_mem_free(amp_c, &status);
    oskar_mem_free(weight_c, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError, "Imager failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("i", status);
}


static PyObject* make_image(PyObject* self, PyObject* args, PyObject* keywds)
{
    PyArrayObject *uu = 0, *vv = 0, *ww = 0, *amp = 0, *weight = 0, *im = 0;
    int i = 0, status = 0, num_vis, num_pixels, size = 0, type = OSKAR_DOUBLE;
    double fov = 0.0, norm = 0.0;
    oskar_Imager* imager;
    oskar_Mem *uu_c, *vv_c, *ww_c, *amp_c, *weight_c, *plane;

    /* Parse inputs. */
    char* keywords[] = {"uu", "vv", "ww", "amp", "weight", "fov", "size", 0};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOOdi", keywords,
            &uu, &vv, &ww, &amp, &weight, &fov, &size))
        return 0;

    /* Check dimensions. */
    if (PyArray_NDIM(uu) != 1 || PyArray_NDIM(vv) != 1 ||
            PyArray_NDIM(ww) != 1 || PyArray_NDIM(amp) != 1 ||
            PyArray_NDIM(weight) != 1)
    {
        PyErr_SetString(PyExc_RuntimeError, "Input data arrays must be 1D.");
        return 0;
    }
    num_vis = (int)*PyArray_DIMS(uu);
    if (num_vis != (int)*PyArray_DIMS(vv) ||
            num_vis != (int)*PyArray_DIMS(ww) ||
            num_vis != (int)*PyArray_DIMS(amp) ||
            num_vis != (int)*PyArray_DIMS(weight))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input data dimension mismatch.");
        return 0;
    }

    /* Create the output image array. */
    num_pixels = size * size;
    npy_intp dims[] = {size, size};
    im = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    /* Create and set up the imager. */
    imager = oskar_imager_create(type, &status);
    oskar_imager_set_fov(imager, fov);
    oskar_imager_set_size(imager, size);

    /* Pointers to input/output arrays. */
    uu_c = oskar_mem_create_alias_from_raw(PyArray_DATA(uu), type,
            OSKAR_CPU, num_vis, &status);
    vv_c = oskar_mem_create_alias_from_raw(PyArray_DATA(vv), type,
            OSKAR_CPU, num_vis, &status);
    ww_c = oskar_mem_create_alias_from_raw(PyArray_DATA(ww), type,
            OSKAR_CPU, num_vis, &status);
    amp_c = oskar_mem_create_alias_from_raw(PyArray_DATA(amp), type,
            OSKAR_CPU, num_vis, &status);
    weight_c = oskar_mem_create_alias_from_raw(PyArray_DATA(weight), type,
            OSKAR_CPU, num_vis, &status);

    /* Make the image. */
    plane = oskar_mem_create(type | OSKAR_COMPLEX, OSKAR_CPU, num_pixels,
            &status);
    oskar_imager_update_plane(imager, num_vis, uu_c, vv_c, ww_c, amp_c,
            weight_c, plane, &norm, &status);
    oskar_imager_finalise_plane(imager, plane, norm, &status);
    memcpy(PyArray_DATA(im), oskar_mem_void_const(plane),
            num_pixels * sizeof(double));

    /* Free memory. */
    oskar_mem_free(uu_c, &status);
    oskar_mem_free(vv_c, &status);
    oskar_mem_free(ww_c, &status);
    oskar_mem_free(amp_c, &status);
    oskar_mem_free(weight_c, &status);
    oskar_imager_free(imager, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError, "Imager failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }

    /* Return image to the python workspace. */
    return Py_BuildValue("O", im);
}

/* Method table. */
static PyMethodDef methods[] =
{
        {"create", (PyCFunction)create, METH_VARARGS,
                "create(type)\n\n"
                "Creates a handle to an OSKAR imager.\n\n"
                "Parameters\n"
                "----------\n"
                "type : string\n"
                "   Either \"double\" or \"single\" to specify "
                "the numerical precision of the images.\n\n"
        },
        {"finalise", (PyCFunction)finalise, METH_VARARGS,
                "finalise(image=None)\n\n"
                "Finalises the image or images and writes them to file.\n\n"
                "Parameters\n"
                "----------\n"
                "image : array like\n"
                "   If given, the output image is returned in this array.\n\n"
        },
        {"make_image", (PyCFunction)make_image, METH_VARARGS | METH_KEYWORDS,
                "make_image(uu, vv, ww, amp, weight, fov, size)\n\n"
                "Makes an image from visibility data.\n\n"
                "Parameters\n"
                "----------\n"
                "uu : array like, shape (n,)\n"
                "   Input baseline u coordinates, in wavelengths.\n\n"
                "vv : array like, shape (n,)\n"
                "   Input baseline v coordinates, in wavelengths.\n\n"
                "ww : array like, shape (n,)\n"
                "   Input baseline w coordinates, in wavelengths.\n\n"
                "amp : array like, shape (n,), complex\n"
                "   Input baseline visibility amplitudes.\n\n"
                "weight : array like, shape (n,)\n"
                "   Input baseline visibility weights.\n\n"
                "fov : scalar\n"
                "   Image field of view, in degrees.\n\n"
                "size : integer\n"
                "   Image size along one dimension, in pixels.\n\n"
        },
        {"reset_cache", (PyCFunction)reset_cache, METH_VARARGS,
                "reset_cache()\n\n"
                "Low-level function to reset the imager's internal memory.\n\n"
        },
        {"run", (PyCFunction)run, METH_VARARGS,
                "run(filename)\n\n"
                "Runs the imager on a visibility file.\n\n"
                "Parameters\n"
                "----------\n"
                "filename : string\n"
                "   Path to input Measurement Set or OSKAR visibility file.\n\n"
        },
        {"set_algorithm", (PyCFunction)set_algorithm, METH_VARARGS,
                "set_algorithm(type)\n\n"
                "Sets the algorithm used by the imager.\n\n"
                "Parameters\n"
                "----------\n"
                "type : string\n"
                "   Either \"FFT\", \"DFT 2D\", \"DFT 3D\".\n\n"
        },
        {"set_default_direction", (PyCFunction)set_default_direction, METH_VARARGS,
                "set_default_direction()\n\n"
                "Clears any direction override.\n\n"
        },
        {"set_direction", (PyCFunction)set_direction, METH_VARARGS,
                "set_direction(ra_deg, dec_deg)\n\n"
                "Sets the image centre to be different to the observation "
                "phase centre.\n\n"
                "Parameters\n"
                "----------\n"
                "ra_deg : scalar\n"
                "   The new image Right Ascension, in degrees.\n\n"
                "dec_deg : scalar\n"
                "   The new image Declination, in degrees.\n\n"
        },
        {"set_channel_range", (PyCFunction)set_channel_range, METH_VARARGS,
                "set_channel_range(start, end, snapshots)\n\n"
                "Sets the visibility channel range used by the imager.\n\n"
                "Parameters\n"
                "----------\n"
                "start : integer\n"
                "   Start channel index.\n\n"
                "end : integer\n"
                "   End channel index (-1 for all channels).\n\n"
                "snapshots : boolean\n"
                "   If true, image each channel separately; "
                "if false, use frequency synthesis.\n\n"
        },
        {"set_grid_kernel", (PyCFunction)set_grid_kernel, METH_VARARGS,
                "set_grid_kernel(type, support, oversample)\n\n"
                "Sets the convolution kernel used for gridding visibilities.\n\n"
                "Parameters\n"
                "----------\n"
                "type : string\n"
                "   \"Spheroidal\".\n\n"
                "support : integer\n"
                "   Support size of kernel (typically 3).\n\n"
                "oversample : integer\n"
                "   Oversampling factor used for look-up table "
                "(typically 100).\n\n"
        },
        {"set_image_type", (PyCFunction)set_image_type, METH_VARARGS,
                "set_image_type(type)\n\n"
                "Sets the image (polarisation) type.\n\n"
                "Parameters\n"
                "----------\n"
                "type : string\n"
                "   Either \"STOKES\", \"I\", \"Q\", \"U\", \"V\","
                "\"LINEAR\", \"XX\", \"XY\", \"YX\", \"YY\" or \"PSF\".\n\n"
        },
        {"set_fov", (PyCFunction)set_fov, METH_VARARGS,
                "set_fov(value)\n\n"
                "Sets the field of view to image.\n\n"
                "Parameters\n"
                "----------\n"
                "value : scalar\n"
                "   Field-of-view, in degrees.\n\n"
        },
        {"set_fft_on_gpu", (PyCFunction)set_fft_on_gpu, METH_VARARGS,
                "set_fft_on_gpu(value)\n\n"
                "Sets whether to use the GPU for FFTs.\n\n"
                "Parameters\n"
                "----------\n"
                "value : integer\n"
                "   If true, use the GPU for FFTs.\n\n"
        },
        {"set_ms_column", (PyCFunction)set_ms_column, METH_VARARGS,
                "set_ms_column(column)\n\n"
                "Sets the data column to use from a Measurement Set.\n\n"
                "Parameters\n"
                "----------\n"
                "column : string\n"
                "   Name of the column to use.\n\n"
        },
        {"set_output_root", (PyCFunction)set_output_root, METH_VARARGS,
                "set_output_root(filename)\n\n"
                "Sets the root path of output images.\n\n"
                "Parameters\n"
                "----------\n"
                "filename : string\n"
                "   Root path.\n\n"
        },
        {"set_size", (PyCFunction)set_size, METH_VARARGS,
                "set_size(value)\n\n"
                "Sets image side length.\n\n"
                "Parameters\n"
                "----------\n"
                "value : integer\n"
                "   Image side length in pixels.\n\n"
        },
        {"set_time_range", (PyCFunction)set_time_range, METH_VARARGS,
                "set_time_range(start, end, snapshots)\n\n"
                "Sets the visibility time range used by the imager.\n\n"
                "Parameters\n"
                "----------\n"
                "start : integer\n"
                "   Start time index.\n\n"
                "end : integer\n"
                "   End time index (-1 for all times).\n\n"
                "snapshots : boolean\n"
                "   If true, image each time slice separately; "
                "if false, use time synthesis.\n\n"
        },
        {"set_vis_frequency", (PyCFunction)set_vis_frequency, METH_VARARGS,
                "set_vis_frequency(ref_hz, inc_hz, num_channels)\n\n"
                "Sets the visibility start frequency.\n\n"
                "Parameters\n"
                "----------\n"
                "ref_hz : scalar\n"
                "   Frequency of index 0, in Hz.\n\n"
                "inc_hz : scalar\n"
                "   Frequency increment, in Hz.\n\n"
                "num_channels : integer\n"
                "   Number of channels in visibility data.\n\n"
        },
        {"set_vis_phase_centre", (PyCFunction)set_vis_phase_centre, METH_VARARGS,
                "set_vis_phase_centre(ra_deg, dec_deg)\n\n"
                "Sets the coordinates of the visibility phase centre.\n\n"
                "Parameters\n"
                "----------\n"
                "ra_deg : scalar\n"
                "   Right Ascension of phase centre, in degrees.\n\n"
                "dec_deg : scalar\n"
                "   Declination of phase centre, in degrees.\n\n"
        },
        {"set_vis_time", (PyCFunction)set_vis_time, METH_VARARGS,
                "set_vis_time(ref_mjd_utc, inc_sec, num_times)\n\n"
                "Sets the visibility start time.\n\n"
                "Parameters\n"
                "----------\n"
                "ref_mjd_utc : scalar\n"
                "   Time of index 0, as MJD(UTC).\n\n"
                "inc_sec : scalar\n"
                "   Time increment, in seconds.\n\n"
                "num_times : integer\n"
                "   Number of time steps in visibility data.\n\n"
        },
        {"update", (PyCFunction)update, METH_VARARGS | METH_KEYWORDS,
                "update(num_baselines, uu, vv, ww, amps, weight, "
                "num_pols, start_time, end_time, start_chan, end_chan)\n\n"
                "Runs the imager using data in memory, and applies "
                "visibility selection.\n\n"
                "The visibility amplitude data dimension order must be:\n"
                "(slowest) time, channel, baseline, polarisation (fastest).\n\n"
                "Parameters\n"
                "----------\n"
                "num_baselines : integer\n"
                "   Number of baselines in the visibility block.\n\n"
                "uu : array like (n,)\n"
                "   Time-baseline ordered U-coordinates, in metres.\n\n"
                "vv : array like (n,)\n"
                "   Time-baseline ordered V-coordinates, in metres.\n\n"
                "ww : array like (n,)\n"
                "   Time-baseline ordered W-coordinates, in metres.\n\n"
                "amps : array like (n,)\n"
                "   Complex visibility amplitudes.\n\n"
                "weight : array like (n,)\n"
                "   Visibility weights.\n\n"
                "num_pols (default 1) : integer\n"
                "   Number of polarisations in the visibility block.\n\n"
                "start_time (default 0) : integer\n"
                "   Start time index of the visibility block.\n\n"
                "end_time (default 0) : integer\n"
                "   End time index of the visibility block.\n\n"
                "start_chan (default 0) : integer\n"
                "   Start channel index of the visibility block.\n\n"
                "end_chan (default 0) : integer\n"
                "   End channel index of the visibility block.\n\n"
        },
        {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_imager_lib",      /* m_name */
        NULL,               /* m_doc */
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
    m = Py_InitModule3("_imager_lib", methods, "docstring ...");
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

