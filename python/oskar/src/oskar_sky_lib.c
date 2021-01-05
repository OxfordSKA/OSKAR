/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <Python.h>

#include <oskar.h>
#include <oskar_version.h>
#include <string.h>

/* http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static const char module_doc[] =
        "This module provides an interface to the OSKAR sky model.";
static const char name[] = "oskar_Sky";

#define deg2rad 1.74532925199432957692369e-2
#define arcsec2rad 4.84813681109535993589914e-6
#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

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


static void sky_free(PyObject* capsule)
{
    int status = 0;
    oskar_sky_free((oskar_Sky*) get_handle(capsule, name), &status);
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


static PyObject* append(PyObject* self, PyObject* args)
{
    oskar_Sky *h1 = 0, *h2 = 0;
    PyObject *capsule1 = 0, *capsule2 = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "OO", &capsule1, &capsule2)) return 0;
    if (!(h1 = (oskar_Sky*) get_handle(capsule1, name))) return 0;
    if (!(h2 = (oskar_Sky*) get_handle(capsule2, name))) return 0;

    /* Append the sky model. */
    oskar_sky_append(h1, h2, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_sky_append() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* append_sources(PyObject* self, PyObject* args)
{
    oskar_Sky *h = 0;
    PyObject *obj[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    oskar_Mem *ra_c, *dec_c, *I_c, *Q_c, *U_c, *V_c;
    oskar_Mem *ref_c, *spix_c, *rm_c, *maj_c, *min_c, *pa_c;
    PyArrayObject *ra = 0, *dec = 0, *I = 0, *Q = 0, *U = 0, *V = 0;
    PyArrayObject *ref = 0, *spix = 0, *rm = 0, *maj = 0, *min = 0, *pa = 0;
    int status = 0, npy_type, type, flags, num_sources, old_num;

    /* Parse inputs: RA, Dec, I, Q, U, V, ref, spix, rm, maj, min, pa. */
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOO", &obj[0],
            &obj[1], &obj[2], &obj[3], &obj[4], &obj[5], &obj[6],
            &obj[7], &obj[8], &obj[9], &obj[10], &obj[11], &obj[12]))
        return 0;
    if (!(h = (oskar_Sky*) get_handle(obj[0], name))) return 0;

    /* Make sure input objects are arrays. Convert if required. */
    flags = NPY_ARRAY_FORCECAST | NPY_ARRAY_IN_ARRAY;
    type = oskar_sky_precision(h);
    npy_type = numpy_type_from_oskar(type);
    ra   = (PyArrayObject*) PyArray_FROM_OTF(obj[1], npy_type, flags);
    dec  = (PyArrayObject*) PyArray_FROM_OTF(obj[2], npy_type, flags);
    I    = (PyArrayObject*) PyArray_FROM_OTF(obj[3], npy_type, flags);
    Q    = (PyArrayObject*) PyArray_FROM_OTF(obj[4], npy_type, flags);
    U    = (PyArrayObject*) PyArray_FROM_OTF(obj[5], npy_type, flags);
    V    = (PyArrayObject*) PyArray_FROM_OTF(obj[6], npy_type, flags);
    ref  = (PyArrayObject*) PyArray_FROM_OTF(obj[7], npy_type, flags);
    spix = (PyArrayObject*) PyArray_FROM_OTF(obj[8], npy_type, flags);
    rm   = (PyArrayObject*) PyArray_FROM_OTF(obj[9], npy_type, flags);
    maj  = (PyArrayObject*) PyArray_FROM_OTF(obj[10], npy_type, flags);
    min  = (PyArrayObject*) PyArray_FROM_OTF(obj[11], npy_type, flags);
    pa   = (PyArrayObject*) PyArray_FROM_OTF(obj[12], npy_type, flags);
    if (!ra || !dec || !I || !Q || !U || !V ||
            !ref || !spix || !rm || !maj || !min || !pa)
        goto fail;

    /* Check size of input arrays. */
    num_sources = (int) PyArray_SIZE(I);
    if (num_sources != (int) PyArray_SIZE(ra) ||
            num_sources != (int) PyArray_SIZE(dec) ||
            num_sources != (int) PyArray_SIZE(Q) ||
            num_sources != (int) PyArray_SIZE(U) ||
            num_sources != (int) PyArray_SIZE(V) ||
            num_sources != (int) PyArray_SIZE(ref) ||
            num_sources != (int) PyArray_SIZE(spix) ||
            num_sources != (int) PyArray_SIZE(rm) ||
            num_sources != (int) PyArray_SIZE(maj) ||
            num_sources != (int) PyArray_SIZE(min) ||
            num_sources != (int) PyArray_SIZE(pa))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input data dimension mismatch.");
        goto fail;
    }

    /* Pointers to input arrays. */
    ra_c = oskar_mem_create_alias_from_raw(PyArray_DATA(ra),
            type, OSKAR_CPU, num_sources, &status);
    dec_c = oskar_mem_create_alias_from_raw(PyArray_DATA(dec),
            type, OSKAR_CPU, num_sources, &status);
    I_c = oskar_mem_create_alias_from_raw(PyArray_DATA(I),
            type, OSKAR_CPU, num_sources, &status);
    Q_c = oskar_mem_create_alias_from_raw(PyArray_DATA(Q),
            type, OSKAR_CPU, num_sources, &status);
    U_c = oskar_mem_create_alias_from_raw(PyArray_DATA(U),
            type, OSKAR_CPU, num_sources, &status);
    V_c = oskar_mem_create_alias_from_raw(PyArray_DATA(V),
            type, OSKAR_CPU, num_sources, &status);
    ref_c = oskar_mem_create_alias_from_raw(PyArray_DATA(ref),
            type, OSKAR_CPU, num_sources, &status);
    spix_c = oskar_mem_create_alias_from_raw(PyArray_DATA(spix),
            type, OSKAR_CPU, num_sources, &status);
    rm_c = oskar_mem_create_alias_from_raw(PyArray_DATA(rm),
            type, OSKAR_CPU, num_sources, &status);
    maj_c = oskar_mem_create_alias_from_raw(PyArray_DATA(maj),
            type, OSKAR_CPU, num_sources, &status);
    min_c = oskar_mem_create_alias_from_raw(PyArray_DATA(min),
            type, OSKAR_CPU, num_sources, &status);
    pa_c = oskar_mem_create_alias_from_raw(PyArray_DATA(pa),
            type, OSKAR_CPU, num_sources, &status);

    /* Copy source data into the sky model. */
    old_num = oskar_sky_num_sources(h);
    oskar_sky_resize(h, old_num + num_sources, &status);
    oskar_mem_copy_contents(oskar_sky_ra_rad(h), ra_c,
            old_num, 0, num_sources, &status);
    oskar_mem_copy_contents(oskar_sky_dec_rad(h), dec_c,
            old_num, 0, num_sources, &status);
    oskar_mem_copy_contents(oskar_sky_I(h), I_c,
            old_num, 0, num_sources, &status);
    oskar_mem_copy_contents(oskar_sky_Q(h), Q_c,
            old_num, 0, num_sources, &status);
    oskar_mem_copy_contents(oskar_sky_U(h), U_c,
            old_num, 0, num_sources, &status);
    oskar_mem_copy_contents(oskar_sky_V(h), V_c,
            old_num, 0, num_sources, &status);
    oskar_mem_copy_contents(oskar_sky_reference_freq_hz(h), ref_c,
            old_num, 0, num_sources, &status);
    oskar_mem_copy_contents(oskar_sky_spectral_index(h), spix_c,
            old_num, 0, num_sources, &status);
    oskar_mem_copy_contents(oskar_sky_rotation_measure_rad(h), rm_c,
            old_num, 0, num_sources, &status);
    oskar_mem_copy_contents(oskar_sky_fwhm_major_rad(h), maj_c,
            old_num, 0, num_sources, &status);
    oskar_mem_copy_contents(oskar_sky_fwhm_minor_rad(h), min_c,
            old_num, 0, num_sources, &status);
    oskar_mem_copy_contents(oskar_sky_position_angle_rad(h), pa_c,
            old_num, 0, num_sources, &status);

    /* Free memory. */
    oskar_mem_free(ra_c, &status);
    oskar_mem_free(dec_c, &status);
    oskar_mem_free(I_c, &status);
    oskar_mem_free(Q_c, &status);
    oskar_mem_free(U_c, &status);
    oskar_mem_free(V_c, &status);
    oskar_mem_free(ref_c, &status);
    oskar_mem_free(spix_c, &status);
    oskar_mem_free(rm_c, &status);
    oskar_mem_free(maj_c, &status);
    oskar_mem_free(min_c, &status);
    oskar_mem_free(pa_c, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "Sky model append_sources() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        goto fail;
    }

    Py_XDECREF(ra);
    Py_XDECREF(dec);
    Py_XDECREF(I);
    Py_XDECREF(Q);
    Py_XDECREF(U);
    Py_XDECREF(V);
    Py_XDECREF(ref);
    Py_XDECREF(spix);
    Py_XDECREF(rm);
    Py_XDECREF(maj);
    Py_XDECREF(min);
    Py_XDECREF(pa);
    return Py_BuildValue("");

fail:
    Py_XDECREF(ra);
    Py_XDECREF(dec);
    Py_XDECREF(I);
    Py_XDECREF(Q);
    Py_XDECREF(U);
    Py_XDECREF(V);
    Py_XDECREF(ref);
    Py_XDECREF(spix);
    Py_XDECREF(rm);
    Py_XDECREF(maj);
    Py_XDECREF(min);
    Py_XDECREF(pa);
    return 0;
}


static PyObject* append_file(PyObject* self, PyObject* args)
{
    oskar_Sky *h = 0, *temp = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* filename = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &filename)) return 0;
    if (!(h = (oskar_Sky*) get_handle(capsule, name))) return 0;

    /* Load the sky model. */
    temp = oskar_sky_load(filename, oskar_sky_precision(h), &status);
    oskar_sky_append(h, temp, &status);
    oskar_sky_free(temp, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_sky_load() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
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
    oskar_Sky* h = 0;
    PyObject* capsule = 0;
    int status = 0, prec = 0;
    const char* type;
    if (!PyArg_ParseTuple(args, "s", &type)) return 0;
    prec = (type[0] == 'S' || type[0] == 's') ? OSKAR_SINGLE : OSKAR_DOUBLE;
    h = oskar_sky_create(prec, OSKAR_CPU, 0, &status);
    capsule = PyCapsule_New((void*)h, name, (PyCapsule_Destructor)sky_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* create_copy(PyObject* self, PyObject* args)
{
    oskar_Sky *h = 0, *t = 0;
    PyObject* capsule = 0;
    int status = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Sky*) get_handle(capsule, name))) return 0;
    t = oskar_sky_create_copy(h, OSKAR_CPU, &status);

    /* Check for errors. */
    if (status || !t)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_sky_create_copy() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        oskar_sky_free(t, &status);
        return 0;
    }

    capsule = PyCapsule_New((void*)t, name, (PyCapsule_Destructor)sky_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* filter_by_flux(PyObject* self, PyObject* args)
{
    oskar_Sky *h = 0;
    PyObject* capsule = 0;
    int status = 0;
    double min_flux_jy = 0.0, max_flux_jy = 0.0;
    if (!PyArg_ParseTuple(args, "Odd", &capsule, &min_flux_jy, &max_flux_jy))
        return 0;
    if (!(h = (oskar_Sky*) get_handle(capsule, name))) return 0;

    /* Filter the sky model. */
    oskar_sky_filter_by_flux(h, min_flux_jy, max_flux_jy, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_sky_filter_by_flux() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* filter_by_radius(PyObject* self, PyObject* args)
{
    oskar_Sky *h = 0;
    PyObject* capsule = 0;
    int status = 0;
    double inner_radius_rad = 0.0, outer_radius_rad = 0.0;
    double ra0_rad = 0.0, dec0_rad = 0.0;
    if (!PyArg_ParseTuple(args, "Odddd", &capsule,
            &inner_radius_rad, &outer_radius_rad, &ra0_rad, &dec0_rad))
        return 0;
    if (!(h = (oskar_Sky*) get_handle(capsule, name))) return 0;

    /* Filter the sky model. */
    oskar_sky_filter_by_radius(h, inner_radius_rad, outer_radius_rad,
            ra0_rad, dec0_rad, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_sky_filter_by_radius() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}

static PyObject* from_array(PyObject* self, PyObject* args)
{
    oskar_Sky* h = 0;
    PyObject *array_object = 0, *capsule = 0;
    PyArrayObject* array = 0;
    npy_intp* dims = 0;
    const char* type = 0;
    int status = 0, flags, i, num_dims, num_columns, num_sources = 0, prec;
    double par[12];
    if (!PyArg_ParseTuple(args, "Os", &array_object, &type)) return 0;

    /* Get a handle to the array and its dimensions. */
    flags = NPY_ARRAY_FORCECAST | NPY_ARRAY_IN_ARRAY;
    array = (PyArrayObject*) PyArray_FROM_OTF(array_object, NPY_DOUBLE, flags);
    if (!array) goto fail;
    num_dims = PyArray_NDIM(array);
    dims = PyArray_DIMS(array);

    /* Check dimensions. */
    if (num_dims > 2)
    {
        PyErr_SetString(PyExc_RuntimeError, "Array has too many dimensions.");
        goto fail;
    }
    num_columns = (num_dims == 2) ? (int) dims[1] : (int) dims[0];
    num_sources = (num_dims == 2) ? (int) dims[0] : 1;
    if (num_columns < 3)
    {
        PyErr_SetString(PyExc_RuntimeError,
                "Must specify at least RA, Dec and Stokes I values.");
        goto fail;
    }
    if (num_columns > 12)
    {
        PyErr_SetString(PyExc_RuntimeError, "Too many source parameters.");
        goto fail;
    }

    /* Create the sky model. */
    prec = (type[0] == 'S' || type[0] == 's') ? OSKAR_SINGLE : OSKAR_DOUBLE;
    h = oskar_sky_create(prec, OSKAR_CPU, 0, &status);

    /* Set the size of the sky model. */
    oskar_sky_resize(h, num_sources, &status);
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_sky_resize() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        goto fail;
    }
    if (num_dims == 1)
    {
        memset(par, 0, sizeof(par));
        memcpy(par, PyArray_DATA(array), num_columns * sizeof(double));
        oskar_sky_set_source(h, 0, par[0] * deg2rad,
                par[1] * deg2rad, par[2], par[3], par[4], par[5],
                par[6], par[7], par[8], par[9] * arcsec2rad,
                par[10] * arcsec2rad, par[11] * deg2rad, &status);
    }
    else
    {
        for (i = 0; i < num_sources; ++i)
        {
            memset(par, 0, sizeof(par));
            memcpy(par, PyArray_GETPTR2(array, i, 0),
                    num_columns * sizeof(double));
            oskar_sky_set_source(h, i, par[0] * deg2rad,
                    par[1] * deg2rad, par[2], par[3], par[4], par[5],
                    par[6], par[7], par[8], par[9] * arcsec2rad,
                    par[10] * arcsec2rad, par[11] * deg2rad, &status);
            if (status) break;
        }
    }
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_sky_set_source() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        goto fail;
    }

    Py_DECREF(array);
    capsule = PyCapsule_New((void*)h, name, (PyCapsule_Destructor)sky_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */

fail:
    oskar_sky_free(h, &status);
    Py_XDECREF(array);
    return 0;
}


static PyObject* from_fits_file(PyObject* self, PyObject* args)
{
    oskar_Sky* h = 0;
    PyObject* capsule = 0;
    int status = 0, override_units = 0, prec = 0;
    double frequency_hz, spectral_index, min_peak_fraction, min_abs_val;
    const char *default_map_units = 0, *filename = 0, *type = 0;
    if (!PyArg_ParseTuple(args, "sddsidds", &filename,
            &min_peak_fraction, &min_abs_val, &default_map_units,
            &override_units, &frequency_hz, &spectral_index, &type))
        return 0;
    prec = (type[0] == 'S' || type[0] == 's') ? OSKAR_SINGLE : OSKAR_DOUBLE;
    h = oskar_sky_from_fits_file(prec, filename, min_peak_fraction,
            min_abs_val, default_map_units, override_units, frequency_hz,
            spectral_index, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_sky_from_fits_file() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        oskar_sky_free(h, &status);
        return 0;
    }
    capsule = PyCapsule_New((void*)h, name, (PyCapsule_Destructor)sky_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* generate_grid(PyObject* self, PyObject* args)
{
    oskar_Sky *h = 0;
    PyObject* capsule = 0;
    int prec, side_length = 0, seed = 1, status = 0;
    const char* type = 0;
    double ra0, dec0, fov, mean_flux_jy, std_flux_jy;
    if (!PyArg_ParseTuple(args, "ddidddis", &ra0, &dec0, &side_length,
            &fov, &mean_flux_jy, &std_flux_jy, &seed, &type)) return 0;

    /* Generate the grid. */
    prec = (type[0] == 'S' || type[0] == 's') ? OSKAR_SINGLE : OSKAR_DOUBLE;
    ra0 *= M_PI / 180.0;
    dec0 *= M_PI / 180.0;
    fov *= M_PI / 180.0;
    h = oskar_sky_generate_grid(prec, ra0, dec0, side_length, fov,
            mean_flux_jy, std_flux_jy, seed, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_sky_generate_grid() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        oskar_sky_free(h, &status);
        return 0;
    }
    capsule = PyCapsule_New((void*)h, name, (PyCapsule_Destructor)sky_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* generate_random_power_law(PyObject* self, PyObject* args)
{
    oskar_Sky *h = 0;
    PyObject* capsule = 0;
    int prec, num_sources = 0, seed = 1, status = 0;
    const char* type = 0;
    double min_flux_jy = 0.0, max_flux_jy = 0.0, power = 0.0;
    if (!PyArg_ParseTuple(args, "idddis", &num_sources, &min_flux_jy,
            &max_flux_jy, &power, &seed, &type)) return 0;

    /* Generate the sources. */
    prec = (type[0] == 'S' || type[0] == 's') ? OSKAR_SINGLE : OSKAR_DOUBLE;
    h = oskar_sky_generate_random_power_law(prec, num_sources,
            min_flux_jy, max_flux_jy, power, seed, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_sky_generate_random_power_law() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        oskar_sky_free(h, &status);
        return 0;
    }
    capsule = PyCapsule_New((void*)h, name, (PyCapsule_Destructor)sky_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* load(PyObject* self, PyObject* args)
{
    oskar_Sky* h = 0;
    PyObject* capsule = 0;
    int status = 0, prec = 0;
    const char *filename = 0, *type = 0;
    if (!PyArg_ParseTuple(args, "ss", &filename, &type)) return 0;
    prec = (type[0] == 'S' || type[0] == 's') ? OSKAR_SINGLE : OSKAR_DOUBLE;
    h = oskar_sky_load(filename, prec, &status);

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_sky_load() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        oskar_sky_free(h, &status);
        return 0;
    }
    capsule = PyCapsule_New((void*)h, name, (PyCapsule_Destructor)sky_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* num_sources(PyObject* self, PyObject* args)
{
    oskar_Sky *h = 0;
    PyObject* capsule = 0;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Sky*) get_handle(capsule, name))) return 0;
    return Py_BuildValue("i", oskar_sky_num_sources(h));
}


static PyObject* save(PyObject* self, PyObject* args)
{
    oskar_Sky *h = 0;
    PyObject* capsule = 0;
    int status = 0;
    const char* filename = 0;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &filename)) return 0;
    if (!(h = (oskar_Sky*) get_handle(capsule, name))) return 0;

    /* Save the sky model. */
#if OSKAR_VERSION >= 0x020800
    oskar_sky_save(h, filename, &status);
#else
    oskar_sky_save(filename, h, &status);
#endif

    /* Check for errors. */
    if (status)
    {
        PyErr_Format(PyExc_RuntimeError,
                "oskar_sky_save() failed with code %d (%s).",
                status, oskar_get_error_string(status));
        return 0;
    }
    return Py_BuildValue("");
}


static PyObject* to_array(PyObject* self, PyObject* args)
{
    oskar_Sky *h = 0;
    PyArrayObject* array1 = 0;
    PyObject* capsule = 0, *array2 = 0;
    npy_intp dims[2];
    int num_sources = 0, type = 0;
    size_t num_bytes;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = (oskar_Sky*) get_handle(capsule, name))) return 0;
    num_sources = oskar_sky_num_sources(h);
    type = oskar_sky_precision(h);

    /* Create a transposed array. */
    dims[0] = 12;
    dims[1] = num_sources;
    array1 = (PyArrayObject*)PyArray_SimpleNew(2, dims,
            type == OSKAR_DOUBLE ? NPY_DOUBLE : NPY_FLOAT);

    /* Copy the data into it. */
    num_bytes = (type == OSKAR_DOUBLE) ? sizeof(double) : sizeof(float);
    num_bytes *= num_sources;
    memcpy(PyArray_GETPTR2(array1, 0, 0),
            oskar_mem_void(oskar_sky_ra_rad(h)), num_bytes);
    memcpy(PyArray_GETPTR2(array1, 1, 0),
            oskar_mem_void(oskar_sky_dec_rad(h)), num_bytes);
    memcpy(PyArray_GETPTR2(array1, 2, 0),
            oskar_mem_void(oskar_sky_I(h)), num_bytes);
    memcpy(PyArray_GETPTR2(array1, 3, 0),
            oskar_mem_void(oskar_sky_Q(h)), num_bytes);
    memcpy(PyArray_GETPTR2(array1, 4, 0),
            oskar_mem_void(oskar_sky_U(h)), num_bytes);
    memcpy(PyArray_GETPTR2(array1, 5, 0),
            oskar_mem_void(oskar_sky_V(h)), num_bytes);
    memcpy(PyArray_GETPTR2(array1, 6, 0),
            oskar_mem_void(oskar_sky_reference_freq_hz(h)), num_bytes);
    memcpy(PyArray_GETPTR2(array1, 7, 0),
            oskar_mem_void(oskar_sky_spectral_index(h)), num_bytes);
    memcpy(PyArray_GETPTR2(array1, 8, 0),
            oskar_mem_void(oskar_sky_rotation_measure_rad(h)), num_bytes);
    memcpy(PyArray_GETPTR2(array1, 9, 0),
            oskar_mem_void(oskar_sky_fwhm_major_rad(h)), num_bytes);
    memcpy(PyArray_GETPTR2(array1, 10, 0),
            oskar_mem_void(oskar_sky_fwhm_minor_rad(h)), num_bytes);
    memcpy(PyArray_GETPTR2(array1, 11, 0),
            oskar_mem_void(oskar_sky_position_angle_rad(h)), num_bytes);

    /* Return a transposed copy. */
    array2 = PyArray_Transpose(array1, 0);
    Py_DECREF(array1);
    return Py_BuildValue("N", array2);
}


/* Method table. */
static PyMethodDef methods[] =
{
        {"append", (PyCFunction)append, METH_VARARGS, "append(sky)"},
        {"append_sources", (PyCFunction)append_sources, METH_VARARGS,
                "append_sources(ra, dec, I, Q, U, V, ref_freq, spectral_index, "
                "rotation_measure, major, minor, position_angle)"},
        {"append_file", (PyCFunction)append_file,
                METH_VARARGS, "append_file(filename)"},
        {"capsule_name", (PyCFunction)capsule_name,
                METH_VARARGS, "capsule_name()"},
        {"create", (PyCFunction)create, METH_VARARGS, "create(precision)"},
        {"create_copy", (PyCFunction)create_copy,
                METH_VARARGS, "create_copy(sky)"},
        {"filter_by_flux", (PyCFunction)filter_by_flux,
                METH_VARARGS, "filter_by_flux(min_flux_jy, max_flux_jy)"},
        {"filter_by_radius", (PyCFunction)filter_by_radius,
                METH_VARARGS, "filter_by_radius(inner_radius_rad, "
                "outer_radius_rad, ra0_rad, dec0_rad)"},
        {"from_array", (PyCFunction)from_array,
                METH_VARARGS, "from_array(array)"},
        {"from_fits_file", (PyCFunction)from_fits_file,
                METH_VARARGS, "from_fits_file(filename, min_peak_fraction, "
                "min_abs_val, default_map_units, override_map_units, "
                "frequency_hz, spectral_index, precision)"},
        {"generate_grid", (PyCFunction)generate_grid,
                METH_VARARGS, "generate_grid(ra0, dec0, side_length, "
                "fov, mean_flux_jy, std_flux_jy, seed, precision)"},
        {"generate_random_power_law", (PyCFunction)generate_random_power_law,
                METH_VARARGS, "generate_random_power_law(num_sources, "
                "min_flux_jy, max_flux_jy, power, seed, precision)"},
        {"load", (PyCFunction)load, METH_VARARGS, "load(filename, precision)"},
        {"num_sources", (PyCFunction)num_sources,
                METH_VARARGS, "num_sources()"},
        {"save", (PyCFunction)save, METH_VARARGS, "save(filename)"},
        {"to_array", (PyCFunction)to_array, METH_VARARGS, "to_array()"},
        {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_sky_lib",         /* m_name */
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
    m = Py_InitModule3("_sky_lib", methods, module_doc);
#endif
    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__sky_lib(void)
{
    import_array();
    return moduleinit();
}
#else
/* The init function name has to match that of the compiled module
 * with the pattern 'init<module name>'. This module is called '_sky_lib' */
PyMODINIT_FUNC init_sky_lib(void)
{
    import_array();
    moduleinit();
    return;
}
#endif

