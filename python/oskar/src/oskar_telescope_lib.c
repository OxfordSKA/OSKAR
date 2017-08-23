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
        "This module provides an interface to the OSKAR telescope model.";
static const char name[] = "oskar_Telescope";

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


static void telescope_free(PyObject* capsule)
{
    int status = 0;
    oskar_telescope_free((oskar_Telescope*) get_handle(capsule, name), &status);
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
    oskar_Telescope* h = 0;
    PyObject* capsule = 0;
    int status = 0, prec = 0;
    const char* type;
    if (!PyArg_ParseTuple(args, "s", &type)) return 0;
    prec = (type[0] == 'S' || type[0] == 's') ? OSKAR_SINGLE : OSKAR_DOUBLE;
    h = oskar_telescope_create(prec, OSKAR_CPU, 0, &status);
    capsule = PyCapsule_New((void*)h, name,
            (PyCapsule_Destructor)telescope_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* load(PyObject* self, PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* dir_name;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &dir_name)) return 0;
    if (!(h = (oskar_Telescope*) get_handle(capsule, name))) return 0;
    oskar_telescope_load(h, dir_name, 0, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_telescope_load() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* identical_stations(PyObject* self, PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Telescope*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("O", oskar_telescope_identical_stations(h) ?
            Py_True : Py_False);
}


static PyObject* max_station_depth(PyObject* self, PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Telescope*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_telescope_max_station_depth(h));
}


static PyObject* max_station_size(PyObject* self, PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Telescope*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_telescope_max_station_size(h));
}


static PyObject* num_baselines(PyObject* self, PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Telescope*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_telescope_num_baselines(h));
}


static PyObject* num_stations(PyObject* self, PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Telescope*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_telescope_num_stations(h));
}


static PyObject* set_allow_station_beam_duplication(PyObject* self,
        PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = (oskar_Telescope*) get_handle(capsule, name))) return 0;
    oskar_telescope_set_allow_station_beam_duplication(h, value);
    return Py_BuildValue("");
}


static PyObject* set_channel_bandwidth(PyObject* self, PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject* capsule = 0;
    double channel_bandwidth_hz = 0.0;
    if (!PyArg_ParseTuple(args, "Od", &capsule,
            &channel_bandwidth_hz)) return 0;
    if (!(h = (oskar_Telescope*) get_handle(capsule, name))) return 0;
    oskar_telescope_set_channel_bandwidth(h, channel_bandwidth_hz);
    return Py_BuildValue("");
}


static PyObject* set_enable_noise(PyObject* self, PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject* capsule = 0;
    int value = 0, seed = 0;
    if (!PyArg_ParseTuple(args, "Oii", &capsule, &value, &seed)) return 0;
    if (!(h = (oskar_Telescope*) get_handle(capsule, name))) return 0;
    oskar_telescope_set_enable_noise(h, value, (unsigned int) seed);
    return Py_BuildValue("");
}


static PyObject* set_enable_numerical_patterns(PyObject* self, PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject* capsule = 0;
    int value = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &value)) return 0;
    if (!(h = (oskar_Telescope*) get_handle(capsule, name))) return 0;
    oskar_telescope_set_enable_numerical_patterns(h, value);
    return Py_BuildValue("");
}


static PyObject* set_gaussian_station_beam_width(PyObject* self,
        PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject* capsule = 0;
    double fwhm_deg = 0.0, ref_freq_hz = 0.0;
    if (!PyArg_ParseTuple(args, "Odd", &capsule, &fwhm_deg, &ref_freq_hz))
        return 0;
    if (!(h = (oskar_Telescope*) get_handle(capsule, name))) return 0;

    /* Check stations exist. */
    if (oskar_telescope_num_stations(h) == 0)
    {
        PyErr_Format(PyExc_RuntimeError, "No stations in telescope model!");
        return 0;
    }
    oskar_telescope_set_gaussian_station_beam_width(h, fwhm_deg, ref_freq_hz);
    return Py_BuildValue("");
}


static PyObject* set_noise_freq(PyObject* self, PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject* capsule = 0;
    int num_channels = 0, status = 0;
    double start_hz = 0.0, inc_hz = 0.0;
    if (!PyArg_ParseTuple(args, "Oddi", &capsule, &start_hz, &inc_hz,
            &num_channels)) return 0;
    if (!(h = (oskar_Telescope*) get_handle(capsule, name))) return 0;

    /* Check stations exist. */
    if (oskar_telescope_num_stations(h) == 0)
    {
        PyErr_Format(PyExc_RuntimeError, "No stations in telescope model!");
        return 0;
    }
    oskar_telescope_set_noise_freq(h, start_hz, inc_hz, num_channels, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_telescope_set_noise_freq() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* set_noise_rms(PyObject* self, PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    double start = 0.0, end = 0.0;
    if (!PyArg_ParseTuple(args, "Odd", &capsule, &start, &end)) return 0;
    if (!(h = (oskar_Telescope*) get_handle(capsule, name))) return 0;

    /* Check stations exist. */
    if (oskar_telescope_num_stations(h) == 0)
    {
        PyErr_Format(PyExc_RuntimeError, "No stations in telescope model!");
        return 0;
    }
    oskar_telescope_set_noise_rms(h, start, end, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_telescope_set_noise_rms() failed "
                "with code %d (%s).\n\n"
                "Remember to set noise frequencies first!",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* set_phase_centre(PyObject* self, PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject* capsule = 0;
    double ra_rad = 0.0, dec_rad = 0.0;
    if (!PyArg_ParseTuple(args, "Odd", &capsule, &ra_rad, &dec_rad)) return 0;
    if (!(h = (oskar_Telescope*) get_handle(capsule, name))) return 0;

    /* Check stations exist. */
    if (oskar_telescope_num_stations(h) == 0)
    {
        PyErr_Format(PyExc_RuntimeError, "No stations in telescope model!");
        return 0;
    }
    oskar_telescope_set_phase_centre(h, OSKAR_SPHERICAL_TYPE_EQUATORIAL,
            ra_rad, dec_rad);
    return Py_BuildValue("");
}


static PyObject* set_pol_mode(PyObject* self, PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* mode;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &mode)) return 0;
    if (!(h = (oskar_Telescope*) get_handle(capsule, name))) return 0;
    oskar_telescope_set_pol_mode(h, mode, &status);
    if (status)
    {
        PyErr_SetString(PyExc_RuntimeError, "Unknown polarisation mode.");
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* set_position(PyObject* self, PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject* capsule = 0;
    double longitude_rad = 0.0, latitude_rad = 0.0, altitude_m = 0.0;
    if (!PyArg_ParseTuple(args, "Oddd", &capsule,
            &longitude_rad, &latitude_rad, &altitude_m)) return 0;
    if (!(h = (oskar_Telescope*) get_handle(capsule, name))) return 0;
    oskar_telescope_set_position(h, longitude_rad, latitude_rad, altitude_m);
    return Py_BuildValue("");
}


static PyObject* set_station_coords_ecef(PyObject* self, PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject *obj[] = {0, 0, 0, 0, 0, 0, 0};
    oskar_Mem *x_c, *y_c, *z_c, *x_err_c, *y_err_c, *z_err_c;
    PyArrayObject *x = 0, *y = 0, *z = 0, *x_err = 0, *y_err = 0, *z_err = 0;
    int status = 0, flags, i, num_stations;
    double longitude, latitude, altitude;

    /* Parse inputs. */
    if (!PyArg_ParseTuple(args, "OdddOOOOOO", &obj[0],
            &longitude, &latitude, &altitude,
            &obj[1], &obj[2], &obj[3], &obj[4], &obj[5], &obj[6])) return 0;
    if (!(h = (oskar_Telescope*) get_handle(obj[0], name))) return 0;

    /* Make sure input objects are arrays. Convert if required. */
    flags = NPY_ARRAY_FORCECAST | NPY_ARRAY_IN_ARRAY;
    x     = (PyArrayObject*) PyArray_FROM_OTF(obj[1], NPY_DOUBLE, flags);
    y     = (PyArrayObject*) PyArray_FROM_OTF(obj[2], NPY_DOUBLE, flags);
    z     = (PyArrayObject*) PyArray_FROM_OTF(obj[3], NPY_DOUBLE, flags);
    x_err = (PyArrayObject*) PyArray_FROM_OTF(obj[4], NPY_DOUBLE, flags);
    y_err = (PyArrayObject*) PyArray_FROM_OTF(obj[5], NPY_DOUBLE, flags);
    z_err = (PyArrayObject*) PyArray_FROM_OTF(obj[6], NPY_DOUBLE, flags);
    if (!x || !y || !z || !x_err || !y_err || !z_err)
        goto fail;

    /* Check size of input arrays. */
    num_stations = (int) PyArray_SIZE(x);
    if (num_stations != (int) PyArray_SIZE(y) ||
            num_stations != (int) PyArray_SIZE(z) ||
            num_stations != (int) PyArray_SIZE(x_err) ||
            num_stations != (int) PyArray_SIZE(y_err) ||
            num_stations != (int) PyArray_SIZE(z_err))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input data dimension mismatch.");
        goto fail;
    }

    /* Pointers to input arrays. */
    x_c = oskar_mem_create_alias_from_raw(PyArray_DATA(x),
            OSKAR_DOUBLE, OSKAR_CPU, num_stations, &status);
    y_c = oskar_mem_create_alias_from_raw(PyArray_DATA(y),
            OSKAR_DOUBLE, OSKAR_CPU, num_stations, &status);
    z_c = oskar_mem_create_alias_from_raw(PyArray_DATA(z),
            OSKAR_DOUBLE, OSKAR_CPU, num_stations, &status);
    x_err_c = oskar_mem_create_alias_from_raw(PyArray_DATA(x_err),
            OSKAR_DOUBLE, OSKAR_CPU, num_stations, &status);
    y_err_c = oskar_mem_create_alias_from_raw(PyArray_DATA(y_err),
            OSKAR_DOUBLE, OSKAR_CPU, num_stations, &status);
    z_err_c = oskar_mem_create_alias_from_raw(PyArray_DATA(z_err),
            OSKAR_DOUBLE, OSKAR_CPU, num_stations, &status);

    /* Set data. */
    oskar_telescope_set_station_coords_ecef(h, longitude, latitude, altitude,
            num_stations, x_c, y_c, z_c, x_err_c, y_err_c, z_err_c, &status);
    for (i = 0; i < num_stations; ++i)
    {
        oskar_Station* station = oskar_telescope_station(h, i);
        oskar_station_resize(station, 1, &status);
        oskar_station_resize_element_types(station, 1, &status);
    }

    /* Free memory. */
    oskar_mem_free(x_c, &status);
    oskar_mem_free(y_c, &status);
    oskar_mem_free(z_c, &status);
    oskar_mem_free(x_err_c, &status);
    oskar_mem_free(y_err_c, &status);
    oskar_mem_free(z_err_c, &status);

    Py_XDECREF(x);
    Py_XDECREF(y);
    Py_XDECREF(z);
    Py_XDECREF(x_err);
    Py_XDECREF(y_err);
    Py_XDECREF(z_err);
    return Py_BuildValue("");

fail:
    Py_XDECREF(x);
    Py_XDECREF(y);
    Py_XDECREF(z);
    Py_XDECREF(x_err);
    Py_XDECREF(y_err);
    Py_XDECREF(z_err);
    return 0;
}


static PyObject* set_station_coords_enu(PyObject* self, PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject *obj[] = {0, 0, 0, 0, 0, 0, 0};
    oskar_Mem *x_c, *y_c, *z_c, *x_err_c, *y_err_c, *z_err_c;
    PyArrayObject *x = 0, *y = 0, *z = 0, *x_err = 0, *y_err = 0, *z_err = 0;
    int status = 0, flags, i, num_stations;
    double longitude, latitude, altitude;

    /* Parse inputs. */
    if (!PyArg_ParseTuple(args, "OdddOOOOOO", &obj[0],
            &longitude, &latitude, &altitude,
            &obj[1], &obj[2], &obj[3], &obj[4], &obj[5], &obj[6])) return 0;
    if (!(h = (oskar_Telescope*) get_handle(obj[0], name))) return 0;

    /* Make sure input objects are arrays. Convert if required. */
    flags = NPY_ARRAY_FORCECAST | NPY_ARRAY_IN_ARRAY;
    x     = (PyArrayObject*) PyArray_FROM_OTF(obj[1], NPY_DOUBLE, flags);
    y     = (PyArrayObject*) PyArray_FROM_OTF(obj[2], NPY_DOUBLE, flags);
    z     = (PyArrayObject*) PyArray_FROM_OTF(obj[3], NPY_DOUBLE, flags);
    x_err = (PyArrayObject*) PyArray_FROM_OTF(obj[4], NPY_DOUBLE, flags);
    y_err = (PyArrayObject*) PyArray_FROM_OTF(obj[5], NPY_DOUBLE, flags);
    z_err = (PyArrayObject*) PyArray_FROM_OTF(obj[6], NPY_DOUBLE, flags);
    if (!x || !y || !z || !x_err || !y_err || !z_err)
        goto fail;

    /* Check size of input arrays. */
    num_stations = (int) PyArray_SIZE(x);
    if (num_stations != (int) PyArray_SIZE(y) ||
            num_stations != (int) PyArray_SIZE(z) ||
            num_stations != (int) PyArray_SIZE(x_err) ||
            num_stations != (int) PyArray_SIZE(y_err) ||
            num_stations != (int) PyArray_SIZE(z_err))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input data dimension mismatch.");
        goto fail;
    }

    /* Pointers to input arrays. */
    x_c = oskar_mem_create_alias_from_raw(PyArray_DATA(x),
            OSKAR_DOUBLE, OSKAR_CPU, num_stations, &status);
    y_c = oskar_mem_create_alias_from_raw(PyArray_DATA(y),
            OSKAR_DOUBLE, OSKAR_CPU, num_stations, &status);
    z_c = oskar_mem_create_alias_from_raw(PyArray_DATA(z),
            OSKAR_DOUBLE, OSKAR_CPU, num_stations, &status);
    x_err_c = oskar_mem_create_alias_from_raw(PyArray_DATA(x_err),
            OSKAR_DOUBLE, OSKAR_CPU, num_stations, &status);
    y_err_c = oskar_mem_create_alias_from_raw(PyArray_DATA(y_err),
            OSKAR_DOUBLE, OSKAR_CPU, num_stations, &status);
    z_err_c = oskar_mem_create_alias_from_raw(PyArray_DATA(z_err),
            OSKAR_DOUBLE, OSKAR_CPU, num_stations, &status);

    /* Set data. */
    oskar_telescope_set_station_coords_enu(h, longitude, latitude, altitude,
            num_stations, x_c, y_c, z_c, x_err_c, y_err_c, z_err_c, &status);
    for (i = 0; i < num_stations; ++i)
    {
        oskar_Station* station = oskar_telescope_station(h, i);
        oskar_station_resize(station, 1, &status);
        oskar_station_resize_element_types(station, 1, &status);
    }

    /* Free memory. */
    oskar_mem_free(x_c, &status);
    oskar_mem_free(y_c, &status);
    oskar_mem_free(z_c, &status);
    oskar_mem_free(x_err_c, &status);
    oskar_mem_free(y_err_c, &status);
    oskar_mem_free(z_err_c, &status);

    Py_XDECREF(x);
    Py_XDECREF(y);
    Py_XDECREF(z);
    Py_XDECREF(x_err);
    Py_XDECREF(y_err);
    Py_XDECREF(z_err);
    return Py_BuildValue("");

fail:
    Py_XDECREF(x);
    Py_XDECREF(y);
    Py_XDECREF(z);
    Py_XDECREF(x_err);
    Py_XDECREF(y_err);
    Py_XDECREF(z_err);
    return 0;
}


static PyObject* set_station_coords_wgs84(PyObject* self, PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject *obj[] = {0, 0, 0, 0};
    oskar_Mem *lon_deg_c, *lat_deg_c, *alt_m_c;
    PyArrayObject *lon_deg = 0, *lat_deg = 0, *alt_m = 0;
    int status = 0, flags, i, num_stations;
    double longitude, latitude, altitude;

    /* Parse inputs. */
    if (!PyArg_ParseTuple(args, "OdddOOO", &obj[0],
            &longitude, &latitude, &altitude,
            &obj[1], &obj[2], &obj[3])) return 0;
    if (!(h = (oskar_Telescope*) get_handle(obj[0], name))) return 0;

    /* Make sure input objects are arrays. Convert if required. */
    flags = NPY_ARRAY_FORCECAST | NPY_ARRAY_IN_ARRAY;
    lon_deg = (PyArrayObject*) PyArray_FROM_OTF(obj[1], NPY_DOUBLE, flags);
    lat_deg = (PyArrayObject*) PyArray_FROM_OTF(obj[2], NPY_DOUBLE, flags);
    alt_m   = (PyArrayObject*) PyArray_FROM_OTF(obj[3], NPY_DOUBLE, flags);
    if (!lon_deg || !lat_deg || !alt_m)
        goto fail;

    /* Check size of input arrays. */
    num_stations = (int) PyArray_SIZE(lon_deg);
    if (num_stations != (int) PyArray_SIZE(lat_deg) ||
            num_stations != (int) PyArray_SIZE(alt_m))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input data dimension mismatch.");
        goto fail;
    }

    /* Pointers to input arrays. */
    lon_deg_c = oskar_mem_create_alias_from_raw(PyArray_DATA(lon_deg),
            OSKAR_DOUBLE, OSKAR_CPU, num_stations, &status);
    lat_deg_c = oskar_mem_create_alias_from_raw(PyArray_DATA(lat_deg),
            OSKAR_DOUBLE, OSKAR_CPU, num_stations, &status);
    alt_m_c = oskar_mem_create_alias_from_raw(PyArray_DATA(alt_m),
            OSKAR_DOUBLE, OSKAR_CPU, num_stations, &status);

    /* Set data. */
    oskar_telescope_set_station_coords_wgs84(h, longitude, latitude, altitude,
            num_stations, lon_deg_c, lat_deg_c, alt_m_c, &status);
    for (i = 0; i < num_stations; ++i)
    {
        oskar_Station* station = oskar_telescope_station(h, i);
        oskar_station_resize(station, 1, &status);
        oskar_station_resize_element_types(station, 1, &status);
    }

    /* Free memory. */
    oskar_mem_free(lon_deg_c, &status);
    oskar_mem_free(lat_deg_c, &status);
    oskar_mem_free(alt_m_c, &status);

    Py_XDECREF(lon_deg);
    Py_XDECREF(lat_deg);
    Py_XDECREF(alt_m);
    return Py_BuildValue("");

fail:
    Py_XDECREF(lon_deg);
    Py_XDECREF(lat_deg);
    Py_XDECREF(alt_m);
    return 0;
}


static PyObject* set_station_type(PyObject* self, PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* type_string = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &type_string)) return 0;
    if (!(h = (oskar_Telescope*) get_handle(capsule, name))) return 0;

    /* Check stations exist. */
    if (oskar_telescope_num_stations(h) == 0)
    {
        PyErr_Format(PyExc_RuntimeError, "No stations in telescope model!");
        return 0;
    }
    oskar_telescope_set_station_type(h, type_string, &status);
    if (status)
    {
        PyErr_SetString(PyExc_RuntimeError, "Unknown station type.");
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* set_time_average(PyObject* self, PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject* capsule = 0;
    double time_average_sec = 0.0;
    if (!PyArg_ParseTuple(args, "Od", &capsule, &time_average_sec)) return 0;
    if (!(h = (oskar_Telescope*) get_handle(capsule, name))) return 0;
    oskar_telescope_set_time_average(h, time_average_sec);
    return Py_BuildValue("");
}


static PyObject* set_uv_filter(PyObject* self, PyObject* args)
{
    oskar_Telescope* h = 0;
    PyObject* capsule = 0;
    int status = 0;
    double uv_filter_min = 0.0, uv_filter_max = 0.0;
    const char* units = 0;
    if (!PyArg_ParseTuple(args, "Odds", &capsule,
            &uv_filter_min, &uv_filter_max, &units)) return 0;
    if (!(h = (oskar_Telescope*) get_handle(capsule, name))) return 0;

    oskar_telescope_set_uv_filter(h,
            uv_filter_min, uv_filter_max, units, &status);
    if (status)
    {
        PyErr_SetString(PyExc_RuntimeError, "Unknown units.");
        return 0;
    }
    return Py_BuildValue("");
}


/* Method table. */
static PyMethodDef methods[] =
{
        {"capsule_name", (PyCFunction)capsule_name,
                METH_VARARGS, "capsule_name()"},
        {"create", (PyCFunction)create, METH_VARARGS, "create(type)"},
        {"load", (PyCFunction)load, METH_VARARGS, "load(input_dir)"},
        {"identical_stations", (PyCFunction)identical_stations,
                METH_VARARGS, "identical_stations()"},
        {"max_station_depth", (PyCFunction)max_station_depth,
                METH_VARARGS, "max_station_depth()"},
        {"max_station_size", (PyCFunction)max_station_size,
                METH_VARARGS, "max_station_size()"},
        {"num_baselines", (PyCFunction)num_baselines,
                METH_VARARGS, "num_baselines()"},
        {"num_stations", (PyCFunction)num_stations,
                METH_VARARGS, "num_stations()"},
        {"set_allow_station_beam_duplication",
                (PyCFunction)set_allow_station_beam_duplication,
                METH_VARARGS, "set_allow_station_beam_duplication(value)"},
        {"set_channel_bandwidth", (PyCFunction)set_channel_bandwidth,
                METH_VARARGS, "set_channel_bandwidth(channel_bandwidth_hz)"},
        {"set_enable_noise", (PyCFunction)set_enable_noise,
                METH_VARARGS, "set_enable_noise(value, seed)"},
        {"set_enable_numerical_patterns",
                (PyCFunction)set_enable_numerical_patterns,
                METH_VARARGS, "set_enable_numerical_patterns(value)"},
        {"set_gaussian_station_beam_width",
                (PyCFunction)set_gaussian_station_beam_width, METH_VARARGS,
                "set_gaussian_station_beam_width(fwhm_deg, ref_freq_hz)"},
        {"set_noise_freq", (PyCFunction)set_noise_freq, METH_VARARGS,
                "set_noise_freq(start_freq_hz, inc_hz, num_channels)"},
        {"set_noise_rms", (PyCFunction)set_noise_rms,
                METH_VARARGS, "set_noise_rms(start, end)"},
        {"set_phase_centre", (PyCFunction)set_phase_centre,
                METH_VARARGS, "set_phase_centre(ra_rad, dec_rad)"},
        {"set_pol_mode", (PyCFunction)set_pol_mode,
                METH_VARARGS, "set_pol_mode(type)"},
        {"set_position", (PyCFunction)set_position,
                METH_VARARGS, "set_position(longitude, latitude, altitude)"},
        {"set_station_coords_ecef", (PyCFunction)set_station_coords_ecef,
                METH_VARARGS, "set_station_coords_ecef(longitude, latitude, "
                        "altitude, x, y, z, x_err, y_err, z_err)"},
        {"set_station_coords_enu", (PyCFunction)set_station_coords_enu,
                METH_VARARGS, "set_station_coords_enu(longitude, latitude, "
                        "altitude, x, y, z, x_err, y_err, z_err)"},
        {"set_station_coords_wgs84", (PyCFunction)set_station_coords_wgs84,
                METH_VARARGS, "set_station_coords_wgs84(longitude, latitude, "
                        "altitude, station_longitudes, station_latitudes, "
                        "station_altitudes)"},
        {"set_station_type", (PyCFunction)set_station_type,
                METH_VARARGS, "set_station_type(type_string)"},
        {"set_time_average", (PyCFunction)set_time_average,
                METH_VARARGS, "set_time_average(time_average_sec)"},
        {"set_uv_filter", (PyCFunction)set_uv_filter, METH_VARARGS,
                "set_uv_filter(uv_filter_min, uv_filter_max, units)"},
        {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_telescope_lib",   /* m_name */
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
    m = Py_InitModule3("_telescope_lib", methods, module_doc);
#endif
    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__telescope_lib(void)
{
    import_array();
    return moduleinit();
}
#else
/* The init function name has to match that of the compiled module
 * with the pattern 'init<module name>'. This module is called '_telescope_lib' */
PyMODINIT_FUNC init_telescope_lib(void)
{
    import_array();
    moduleinit();
    return;
}
#endif

