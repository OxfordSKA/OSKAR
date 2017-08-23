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

/* http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html */
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stddef.h>

typedef struct DComplex {
    double re;
    double im;
} DComplex;

typedef struct Complex {
    float re;
    float im;
} Complex;


/**
 * @brief
 * @details
 */
static PyObject* apply_gains(PyObject* self, PyObject* args)
{
    /* Read input arguments */
    PyObject *vis_in_ = 0, *gains_ = 0;
    PyObject *pyo_vis_in = 0, *pyo_vis_out = 0, *pyo_gains = 0;
    if (!PyArg_ParseTuple(args, "OO", &vis_in_, &gains_))
        return 0;

    /* Return an ndarray from the python objects */
    pyo_vis_in = PyArray_FROM_OTF(vis_in_, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY);
    pyo_gains = PyArray_FROM_OTF(gains_, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!pyo_vis_in || !pyo_gains) goto fail;

    /* TODO(BM) Error checking on the dims! */
    int nd_vis = PyArray_NDIM((PyArrayObject*)pyo_vis_in);
    npy_intp* vis_dims = PyArray_DIMS((PyArrayObject*)pyo_vis_in);
    /*int num_vis = (int) vis_dims[0];*/

    /*int nd_gains = PyArray_NDIM((PyArrayObject*)pyo_gains);*/
    npy_intp* gains_dims = PyArray_DIMS((PyArrayObject*)pyo_gains);
    int num_antennas = (int) gains_dims[0];

    /* Create PyObject for output visibilities */
    pyo_vis_out = PyArray_SimpleNew(nd_vis, vis_dims, NPY_CDOUBLE);

    /* Apply the gains: v_out = gp * v_in * conj(gq) */
    DComplex* v_out = (DComplex*)PyArray_DATA((PyArrayObject*)pyo_vis_out);
    DComplex* g = (DComplex*)PyArray_DATA((PyArrayObject*)pyo_gains);
    DComplex* v_in = (DComplex*)PyArray_DATA((PyArrayObject*)pyo_vis_in);
    for (int i = 0, p = 0; p < num_antennas; ++p) {
        for (int q = p + 1; q < num_antennas; ++q, ++i) {
            double a = v_in[i].re * g[p].re - v_in[i].im * g[p].im;
            double b = v_in[i].im * g[p].re + v_in[i].re * g[p].im;
            v_out[i].re = a * g[q].re + b * g[q].im;
            v_out[i].im = b * g[q].re - a * g[q].im;
        }
    }
    /* Decrement references to temporary array objects. */
    Py_XDECREF(pyo_vis_in);
    Py_XDECREF(pyo_gains);
    return Py_BuildValue("N", pyo_vis_out);

    /*
    printf("  - Ref count: %zi, %zi, %zi\n",
            PyArray_REFCOUNT(pyo_vis_in),
            PyArray_REFCOUNT(pyo_gains),
            PyArray_REFCOUNT(pyo_vis_out));
    */

fail:
    Py_XDECREF(pyo_gains);
    Py_XDECREF(pyo_vis_in);
    Py_XDECREF(pyo_vis_out);
    return 0;
}


/**
 * @brief
 * @details
 */
static PyObject* apply_gains_2(PyObject* self, PyObject* args)
{
    /* Read input arguments */
    PyObject *vis_in_o = 0, *gains_o = 0;
    PyObject *pyo_vis_in = 0, *pyo_vis_out = 0, *pyo_gains = 0;
    if (!PyArg_ParseTuple(args, "OO", &vis_in_o, &gains_o))
        return 0;

    /* Return an ndarray from the python objects */
    pyo_vis_in = PyArray_FROM_OTF(vis_in_o, NPY_CFLOAT, NPY_ARRAY_IN_ARRAY);
    pyo_gains = PyArray_FROM_OTF(gains_o, NPY_CFLOAT, NPY_ARRAY_IN_ARRAY);
    if (!pyo_vis_in || !pyo_gains) goto fail;

    /* TODO(BM) Error checking on the dims! */
    int nd_vis = PyArray_NDIM((PyArrayObject*)pyo_vis_in);
    npy_intp* vis_dims = PyArray_DIMS((PyArrayObject*)pyo_vis_in);
    /*int num_vis = (int) vis_dims[0];*/

    /*int nd_gains = PyArray_NDIM((PyArrayObject*)pyo_gains);*/
    npy_intp* gains_dims = PyArray_DIMS((PyArrayObject*)pyo_gains);
    int num_antennas = (int) gains_dims[0];

    /* Create PyObject for output visibilities */
    pyo_vis_out = PyArray_SimpleNew(nd_vis, vis_dims, NPY_CFLOAT);

    /* Apply the gains: v_out = gp * v_in * conj(gq) */
    Complex* v_out = (Complex*)PyArray_DATA((PyArrayObject*)pyo_vis_out);
    Complex* g = (Complex*)PyArray_DATA((PyArrayObject*)pyo_gains);
    Complex* v_in = (Complex*)PyArray_DATA((PyArrayObject*)pyo_vis_in);
    for (int i = 0, p = 0; p < num_antennas; ++p) {
        for (int q = p + 1; q < num_antennas; ++q, ++i) {
            float a = v_in[i].re * g[p].re - v_in[i].im * g[p].im;
            float b = v_in[i].im * g[p].re + v_in[i].re * g[p].im;
            v_out[i].re = a * g[q].re + b * g[q].im;
            v_out[i].im = b * g[q].re - a * g[q].im;
        }
    }
    /* Decrement references to temporary array objects. */
    Py_XDECREF(pyo_vis_in);
    Py_XDECREF(pyo_gains);
    return Py_BuildValue("N", pyo_vis_out);

fail:
    Py_XDECREF(pyo_gains);
    Py_XDECREF(pyo_vis_in);
    Py_XDECREF(pyo_vis_out);
    return 0;
}



/**
 * @brief
 * @details
 */
static PyObject* vis_list_to_matrix(PyObject* self, PyObject* args)
{
    /* Read input arguments */
    PyObject* vis_list_ = NULL;
    int num_ant = 0;
    if (!PyArg_ParseTuple(args, "Oi", &vis_list_, &num_ant))
        return NULL;

    /* Convert to an ndarray */
    PyObject* pyo_vis_list = PyArray_FROM_OTF(vis_list_, NPY_CDOUBLE,
                                              NPY_ARRAY_IN_ARRAY);
    if (!pyo_vis_list) goto fail;

    /* TODO(BM) Error checking on the dims!
    int nd_vis = PyArray_NDIM((PyArrayObject*)pyo_vis_list);
    npy_intp* vis_dims = PyArray_DIMS((PyArrayObject*)pyo_vis_list);
    int num_vis = vis_dims[0];*/

    /* Create PyObject for output visibilities */
    npy_intp dims[] = { num_ant, num_ant };
    PyObject* pyo_vis_matrix = PyArray_SimpleNew(2, dims, NPY_CDOUBLE);

    DComplex* v_list = (DComplex*)PyArray_DATA((PyArrayObject*)pyo_vis_list);
    DComplex* v_matrix = (DComplex*)PyArray_DATA((PyArrayObject*)pyo_vis_matrix);
    for (int i = 0, p = 0; p < num_ant; ++p) {
        for (int q = p + 1; q < num_ant; ++q, ++i) {
            v_matrix[p * num_ant + q].re = v_list[i].re;
            v_matrix[p * num_ant + q].im = -v_list[i].im;
            v_matrix[q * num_ant + p].re = v_list[i].re;
            v_matrix[q * num_ant + p].im = v_list[i].im;
        }
    }
    for (int i = 0; i < num_ant; ++i) {
        v_matrix[i * num_ant + i].re = 0.0;
        v_matrix[i * num_ant + i].im = 0.0;
    }

    /* Decrement references to temporary array objects. */
    Py_XDECREF(pyo_vis_list);
    return Py_BuildValue("N", pyo_vis_matrix);

fail:
    Py_XDECREF(pyo_vis_list);
    return 0;
}


/**
 * @brief
 * @details
 */
static PyObject* vis_list_to_matrix_2(PyObject* self, PyObject* args)
{
    /* Read input arguments */
    PyObject* vis_list_ = NULL;
    int num_ant = 0;
    if (!PyArg_ParseTuple(args, "Oi", &vis_list_, &num_ant))
        return NULL;

    /* Convert to an ndarray */
    PyObject* pyo_vis_list = PyArray_FROM_OTF(vis_list_, NPY_CFLOAT,
                                              NPY_ARRAY_IN_ARRAY);
    if (!pyo_vis_list) goto fail;

    /* TODO(BM) Error checking on the dims!
    int nd_vis = PyArray_NDIM((PyArrayObject*)pyo_vis_list);
    npy_intp* vis_dims = PyArray_DIMS((PyArrayObject*)pyo_vis_list);
    int num_vis = vis_dims[0];*/

    /* Create PyObject for output visibilities */
    npy_intp dims[] = { num_ant, num_ant };
    PyObject* pyo_vis_matrix = PyArray_SimpleNew(2, dims, NPY_CFLOAT);

    Complex* v_list = (Complex*)PyArray_DATA((PyArrayObject*)pyo_vis_list);
    Complex* v_matrix = (Complex*)PyArray_DATA((PyArrayObject*)pyo_vis_matrix);
    for (int i = 0, p = 0; p < num_ant; ++p) {
        for (int q = p + 1; q < num_ant; ++q, ++i) {
            v_matrix[p * num_ant + q].re = v_list[i].re;
            v_matrix[p * num_ant + q].im = -v_list[i].im;
            v_matrix[q * num_ant + p].re = v_list[i].re;
            v_matrix[q * num_ant + p].im = v_list[i].im;
        }
    }
    for (int i = 0; i < num_ant; ++i) {
        v_matrix[i * num_ant + i].re = 0.0;
        v_matrix[i * num_ant + i].im = 0.0;
    }

    /* Decrement references to temporary array objects. */
    Py_XDECREF(pyo_vis_list);
    return Py_BuildValue("N", pyo_vis_matrix);

fail:
    Py_XDECREF(pyo_vis_list);
    return 0;
}



static PyObject* check_ref_count(PyObject* self, PyObject* args)
{
    PyObject* obj = NULL;
    /* https://docs.python.org/2/c-api/arg.html */
    /* Reference count is not increased by 'O' for PyArg_ParseTyple */
    if (!PyArg_ParseTuple(args, "O", &obj))
        return NULL;
    return Py_BuildValue("ii", PyArray_REFCOUNT(obj), Py_REFCNT(obj));
}


static PyObject* expand(PyObject* self, PyObject* args)
{
    PyObject *vis_in_comp = 0, *vis_out_orig = 0;
    PyArrayObject *ant1_, *ant2_, *weight_, *data_in_, *data_out_;
    int num_antennas = 0, num_baselines = 0, num_input_vis = 0;
    int *ant1, *ant2, a1, a2, b, t, row_in, row_out, *out_time_idx = 0, w;
    double *data_in, *data_out, *weight, d[2];
    const char *input_name, *output_name;
    if (!PyArg_ParseTuple(args, "iOsOs", &num_antennas,
            &vis_in_comp, &input_name, &vis_out_orig, &output_name))
        return 0;
    num_baselines = num_antennas * (num_antennas - 1) / 2;
    ant1_     = (PyArrayObject*)PyDict_GetItemString(vis_in_comp, "antenna1");
    ant2_     = (PyArrayObject*)PyDict_GetItemString(vis_in_comp, "antenna2");
    weight_   = (PyArrayObject*)PyDict_GetItemString(vis_in_comp, "weight");
    data_in_  = (PyArrayObject*)PyDict_GetItemString(vis_in_comp, input_name);
    data_out_ = (PyArrayObject*)PyDict_GetItemString(vis_out_orig, output_name);
    num_input_vis = (int)*PyArray_DIMS(data_in_);
    ant1     = (int*) PyArray_DATA(ant1_);
    ant2     = (int*) PyArray_DATA(ant2_);
    weight   = (double*) PyArray_DATA(weight_);
    data_in  = (double*) PyArray_DATA(data_in_);
    data_out = (double*) PyArray_DATA(data_out_);
    out_time_idx = (int*) calloc(num_baselines, sizeof(int));

    for (row_in = 0; row_in < num_input_vis; ++row_in)
    {
        a1 = ant1[row_in];
        a2 = ant2[row_in];
        w = (int) round(weight[row_in]);
        d[0] = data_in[2*row_in + 0];
        d[1] = data_in[2*row_in + 1];
        b = a1 * (num_antennas - 1) - (a1 - 1) * a1 / 2 + a2 - a1 - 1;
        for (t = out_time_idx[b]; t < out_time_idx[b] + w; ++t)
        {
            row_out = t * num_baselines + b;
            data_out[2*row_out + 0] = d[0];
            data_out[2*row_out + 1] = d[1];
        }
        out_time_idx[b] += w;
    }

    free(out_time_idx);
    return Py_BuildValue("");
}

struct oskar_BDA
{
    /* Options. */
    int num_antennas, num_baselines, num_pols, num_times;
    double duvw_max, dt_max, max_fact, fov_deg, delta_t;

    /* Output data. */
    int bda_row, output_size, *ant1, *ant2;
    double *u, *v, *w, *data, *time, *exposure, *sigma, *weight;

    /* Current UVW coordinates. */
    double *uu_current, *vv_current, *ww_current;

    /* Buffers of deltas along the baseline in the current average. */
    double *duvw, *dt;

    /* Buffers of the current average along the baseline. */
    double *ave_uu, *ave_vv, *ave_ww, *ave_data;
    int *ave_count;
};
typedef struct oskar_BDA oskar_BDA;

static const char* name = "oskar_BDA";

#define INT sizeof(int)
#define DBL sizeof(double)

static void oskar_bda_clear(oskar_BDA* h)
{
    /* Clear averages. */
    memset(h->uu_current, '\0', h->num_baselines * DBL);
    memset(h->vv_current, '\0', h->num_baselines * DBL);
    memset(h->ww_current, '\0', h->num_baselines * DBL);
    memset(h->duvw, '\0', h->num_baselines * DBL);
    memset(h->dt, '\0', h->num_baselines * DBL);
    memset(h->ave_uu, '\0', h->num_baselines * DBL);
    memset(h->ave_vv, '\0', h->num_baselines * DBL);
    memset(h->ave_ww, '\0', h->num_baselines * DBL);
    memset(h->ave_count, '\0', h->num_baselines * INT);
    memset(h->ave_data, '\0', h->num_baselines * h->num_pols * 2*DBL);

    /* Free the output data. */
    h->output_size = 0;
    h->bda_row = 0;
    free(h->ant1);
    free(h->ant2);
    free(h->u);
    free(h->v);
    free(h->w);
    free(h->data);
    free(h->time);
    free(h->exposure);
    free(h->sigma);
    free(h->weight);
    h->ant1 = 0;
    h->ant2 = 0;
    h->u = 0;
    h->v = 0;
    h->w = 0;
    h->data = 0;
    h->time = 0;
    h->exposure = 0;
    h->sigma = 0;
    h->weight = 0;
}


static void oskar_bda_free(oskar_BDA* h)
{
    oskar_bda_clear(h);
    free(h->uu_current);
    free(h->vv_current);
    free(h->ww_current);
    free(h->duvw);
    free(h->dt);
    free(h->ave_uu);
    free(h->ave_vv);
    free(h->ave_ww);
    free(h->ave_data);
    free(h->ave_count);
    free(h);
}


/* arcsinc(x) function from Obit. Uses Newton-Raphson method.  */
static double inv_sinc(double value)
{
    double x1 = 0.001;
    for (int i = 0; i < 1000; ++i)
    {
        double x0 = x1;
        double a = x0 * M_PI;
        x1 = x0 - ((sin(a) / a) - value) /
                        ((a * cos(a) - M_PI * sin(a)) / (a * a));
        if (fabs(x1 - x0) < 1.0e-6)
            break;
    }
    return x1;
}


static void bda_free(PyObject* capsule)
{
    oskar_BDA* h = (oskar_BDA*) PyCapsule_GetPointer(capsule, name);
    oskar_bda_free(h);
}


static oskar_BDA* get_handle(PyObject* capsule)
{
    oskar_BDA* h = 0;
    if (!PyCapsule_CheckExact(capsule))
    {
        PyErr_SetString(PyExc_RuntimeError, "Object is not a PyCapsule.");
        return 0;
    }
    h = (oskar_BDA*) PyCapsule_GetPointer(capsule, name);
    if (!h)
    {
        PyErr_SetString(PyExc_RuntimeError,
                "Capsule is not of type oskar_BDA.");
        return 0;
    }
    return h;
}


static PyObject* bda_create(PyObject* self, PyObject* args)
{
    oskar_BDA* h = 0;
    PyObject* capsule = 0;
    int num_antennas, num_baselines, num_pols;
    if (!PyArg_ParseTuple(args, "ii", &num_antennas, &num_pols))
        return 0;

    /* Create and initialise the BDA object. */
    h = (oskar_BDA*) calloc(1, sizeof(oskar_BDA));
    num_baselines = num_antennas * (num_antennas - 1) / 2;
    h->num_antennas  = num_antennas;
    h->num_baselines = num_baselines;
    h->num_pols      = num_pols;
    h->uu_current    = (double*) calloc(num_baselines, DBL);
    h->vv_current    = (double*) calloc(num_baselines, DBL);
    h->ww_current    = (double*) calloc(num_baselines, DBL);
    h->duvw          = (double*) calloc(num_baselines, DBL);
    h->dt            = (double*) calloc(num_baselines, DBL);
    h->ave_uu        = (double*) calloc(num_baselines, DBL);
    h->ave_vv        = (double*) calloc(num_baselines, DBL);
    h->ave_ww        = (double*) calloc(num_baselines, DBL);
    h->ave_count     = (int*)    calloc(num_baselines, INT);
    h->ave_data      = (double*) calloc(num_baselines * num_pols, 2*DBL);

    /* Initialise and return the PyCapsule. */
    capsule = PyCapsule_New((void*)h, name, (PyCapsule_Destructor)bda_free);
    return Py_BuildValue("N", capsule); /* Don't increment refcount. */
}


static PyObject* bda_set_compression(PyObject* self, PyObject* args)
{
    oskar_BDA* h = 0;
    PyObject* capsule = 0;
    double max_fact = 0.0, fov_deg = 0.0, wavelength = 0.0, dt_max = 0.0;
    if (!PyArg_ParseTuple(args, "Odddd",
            &capsule, &max_fact, &fov_deg, &wavelength, &dt_max))
        return 0;
    if (!(h = get_handle(capsule))) return 0;
    h->max_fact = max_fact;
    h->fov_deg = fov_deg;
    h->duvw_max = inv_sinc(1.0 / max_fact) / (fov_deg * (M_PI / 180.0));
    h->duvw_max *= wavelength;
    h->dt_max = dt_max;
    return Py_BuildValue("d", h->duvw_max);
}


static PyObject* bda_set_delta_t(PyObject* self, PyObject* args)
{
    oskar_BDA* h = 0;
    PyObject* capsule = 0;
    double delta_t = 0.0;
    if (!PyArg_ParseTuple(args, "Od", &capsule, &delta_t)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    h->delta_t = delta_t;
    return Py_BuildValue("");
}


static PyObject* bda_set_num_times(PyObject* self, PyObject* args)
{
    oskar_BDA* h = 0;
    PyObject* capsule = 0;
    int num_times = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &num_times)) return 0;
    if (!(h = get_handle(capsule))) return 0;
    h->num_times = num_times;
    return Py_BuildValue("");
}


static PyObject* bda_set_initial_coords(PyObject* self, PyObject* args)
{
    oskar_BDA* h = 0;
    PyObject *obj[] = {0, 0, 0, 0};
    PyArrayObject *uu_ = 0, *vv_ = 0, *ww_ = 0;
    if (!PyArg_ParseTuple(args, "OOOO",
            &obj[0], &obj[1], &obj[2], &obj[3]))
        return 0;
    if (!(h = get_handle(obj[0]))) return 0;

    /* Make sure input objects are arrays. Convert if required. */
    uu_ = (PyArrayObject*) PyArray_FROM_OTF(obj[1],
            NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (!uu_) goto fail;
    vv_ = (PyArrayObject*) PyArray_FROM_OTF(obj[2],
            NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (!vv_) goto fail;
    ww_ = (PyArrayObject*) PyArray_FROM_OTF(obj[3],
            NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (!ww_) goto fail;

    /* Check dimensions. */
    if (PyArray_NDIM(uu_) != 1 || PyArray_NDIM(vv_) != 1 ||
            PyArray_NDIM(ww_) != 1)
    {
        PyErr_SetString(PyExc_RuntimeError, "Coordinate arrays must be 1D.");
        goto fail;
    }
    if (h->num_baselines != (int)*PyArray_DIMS(uu_) ||
            h->num_baselines != (int)*PyArray_DIMS(vv_) ||
            h->num_baselines != (int)*PyArray_DIMS(ww_))
    {
        PyErr_SetString(PyExc_RuntimeError, "Input data dimension mismatch.");
        goto fail;
    }

    /* Copy the data. */
    memcpy(h->uu_current, PyArray_DATA(uu_), h->num_baselines * DBL);
    memcpy(h->vv_current, PyArray_DATA(vv_), h->num_baselines * DBL);
    memcpy(h->ww_current, PyArray_DATA(ww_), h->num_baselines * DBL);

    Py_XDECREF(uu_);
    Py_XDECREF(vv_);
    Py_XDECREF(ww_);
    return Py_BuildValue("");

fail:
    Py_XDECREF(uu_);
    Py_XDECREF(vv_);
    Py_XDECREF(ww_);
    return 0;
}


static PyObject* bda_add_data(PyObject* self, PyObject* args)
{
    oskar_BDA* h = 0;
    PyObject *obj[] = {0, 0, 0, 0, 0};
    PyArrayObject *amp_current_ = 0;
    PyArrayObject *uu_next_ = 0, *vv_next_ = 0, *ww_next_ = 0;
    int a1, a2, b, i, j, p, t = 0;
    double *uu_next = 0, *vv_next = 0, *ww_next = 0, *data;
    if (!PyArg_ParseTuple(args, "OiOOOO", &obj[0], &t, &obj[1],
            &obj[2], &obj[3], &obj[4])) return 0;
    if (!(h = get_handle(obj[0]))) return 0;

    /* Get array handles. */
    amp_current_ = (PyArrayObject*) PyArray_FROM_OTF(obj[1],
            NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (!amp_current_) goto fail;
    if ((obj[2] != Py_None) && (obj[3] != Py_None) && (obj[4] != Py_None))
    {
        /* Make sure input objects are arrays. Convert if required. */
        uu_next_ = (PyArrayObject*) PyArray_FROM_OTF(obj[2],
                NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
        if (!uu_next_) goto fail;
        vv_next_ = (PyArrayObject*) PyArray_FROM_OTF(obj[3],
                NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
        if (!vv_next_) goto fail;
        ww_next_ = (PyArrayObject*) PyArray_FROM_OTF(obj[4],
                NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
        if (!ww_next_) goto fail;

        /* Check dimensions. */
        if (PyArray_NDIM(uu_next_) != 1 ||
                PyArray_NDIM(vv_next_) != 1 ||
                PyArray_NDIM(ww_next_) != 1)
        {
            PyErr_SetString(PyExc_RuntimeError,
                    "Coordinate arrays must be 1D.");
            goto fail;
        }
        if (h->num_baselines != (int)*PyArray_DIMS(uu_next_) ||
                h->num_baselines != (int)*PyArray_DIMS(vv_next_) ||
                h->num_baselines != (int)*PyArray_DIMS(ww_next_))
        {
            PyErr_SetString(PyExc_RuntimeError,
                    "Input data dimension mismatch.");
            goto fail;
        }
    }

    if (uu_next_) uu_next = (double*) PyArray_DATA(uu_next_);
    if (vv_next_) vv_next = (double*) PyArray_DATA(vv_next_);
    if (ww_next_) ww_next = (double*) PyArray_DATA(ww_next_);
    data                  = (double*) PyArray_DATA(amp_current_);
    for (a1 = 0, b = 0; a1 < h->num_antennas; ++a1)
    {
        for (a2 = a1 + 1; a2 < h->num_antennas; ++a2, ++b)
        {
            double du, dv, dw, b_duvw = 0.0, s;

            /* Accumulate into averages. */
            h->ave_count[b] += 1;
            h->ave_uu[b]    += h->uu_current[b];
            h->ave_vv[b]    += h->vv_current[b];
            h->ave_ww[b]    += h->ww_current[b];
            for (p = 0; p < h->num_pols; ++p)
            {
                i = b * h->num_pols + p;
                h->ave_data[2*i + 0] += data[2*i + 0];
                h->ave_data[2*i + 1] += data[2*i + 1];
            }

            /* Look ahead to the next point on the baseline and see if
             * this is also in the average. */
             if ((t < h->num_times - 1) && uu_next && vv_next && ww_next)
             {
                 du = uu_next[b] - h->uu_current[b];
                 dv = vv_next[b] - h->vv_current[b];
                 dw = ww_next[b] - h->ww_current[b];
                 b_duvw = sqrt(du*du + dv*dv + dw*dw);
             }

             /* If last time or if next point extends beyond average,
              * save out averaged data and reset,
              * else accumulate current average lengths. */
             if (t == h->num_times - 1 ||
                     h->duvw[b] + b_duvw > h->duvw_max ||
                     h->dt[b] + h->delta_t > h->dt_max)
             {
                 s = 1.0 / h->ave_count[b];
                 h->ave_uu[b] *= s;
                 h->ave_vv[b] *= s;
                 h->ave_ww[b] *= s;
                 for (p = 0; p < h->num_pols; ++p)
                 {
                     i = b * h->num_pols + p;
                     h->ave_data[2*i + 0] *= s;
                     h->ave_data[2*i + 1] *= s;
                 }

                 /* Store the averaged data. */
                 if (h->output_size < h->bda_row + 1)
                 {
                     h->output_size += h->num_baselines;
                     h->ant1     = realloc(h->ant1, h->output_size * INT);
                     h->ant2     = realloc(h->ant2, h->output_size * INT);
                     h->u        = realloc(h->u, h->output_size * DBL);
                     h->v        = realloc(h->v, h->output_size * DBL);
                     h->w        = realloc(h->w, h->output_size * DBL);
                     h->exposure = realloc(h->exposure, h->output_size * DBL);
                     h->sigma    = realloc(h->sigma, h->output_size * DBL);
                     h->weight   = realloc(h->weight, h->output_size * DBL);
                     h->data     = realloc(h->data,
                             h->num_pols * h->output_size * 2*DBL);
                 }
                 h->ant1[h->bda_row] = a1;
                 h->ant2[h->bda_row] = a2;
                 h->u[h->bda_row] = h->ave_uu[b];
                 h->v[h->bda_row] = h->ave_vv[b];
                 h->w[h->bda_row] = h->ave_ww[b];
                 h->exposure[h->bda_row] = h->ave_count[b] * h->delta_t;
                 h->sigma[h->bda_row] = sqrt(s);
                 h->weight[h->bda_row] = h->ave_count[b];
                 for (p = 0; p < h->num_pols; ++p)
                 {
                     i = b * h->num_pols + p;
                     j = h->bda_row * h->num_pols + p;
                     h->data[2*j + 0] = h->ave_data[2*i + 0];
                     h->data[2*j + 1] = h->ave_data[2*i + 1];
                 }

                 /* Reset baseline accumulation buffers. */
                 h->duvw[b] = 0.0;
                 h->dt[b] = 0.0;
                 h->ave_count[b] = 0;
                 h->ave_uu[b] = 0.0;
                 h->ave_vv[b] = 0.0;
                 h->ave_ww[b] = 0.0;
                 for (p = 0; p < h->num_pols; ++p)
                 {
                     i = b * h->num_pols + p;
                     h->ave_data[2*i + 0] = 0.0;
                     h->ave_data[2*i + 1] = 0.0;
                 }

                 /* Update baseline average row counter for next output. */
                 h->bda_row += 1;
             }
             else
             {
                 /* Accumulate distance and time on current baseline. */
                 h->duvw[b] += b_duvw;
                 h->dt[b] += h->delta_t;
             }
        }
    } /* end baseline loop */

    /* Copy "next" to "current" for the next time block. */
    if (uu_next) memcpy(h->uu_current, uu_next, h->num_baselines * DBL);
    if (vv_next) memcpy(h->vv_current, vv_next, h->num_baselines * DBL);
    if (ww_next) memcpy(h->ww_current, ww_next, h->num_baselines * DBL);

    Py_XDECREF(amp_current_);
    Py_XDECREF(uu_next_);
    Py_XDECREF(vv_next_);
    Py_XDECREF(ww_next_);
    return Py_BuildValue("");

fail:
    Py_XDECREF(amp_current_);
    Py_XDECREF(uu_next_);
    Py_XDECREF(vv_next_);
    Py_XDECREF(ww_next_);
    return 0;
}


static PyObject* bda_finalise(PyObject* self, PyObject* args)
{
    oskar_BDA* h = 0;
    PyObject* capsule = 0, *dict;
    PyArrayObject *ant1, *ant2, *uu, *vv, *ww, *data;
    PyArrayObject *expo, *sigma, *weight;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return 0;
    if (!(h = get_handle(capsule))) return 0;

    /* Create arrays that will be returned to Python. */
    npy_intp dims1[] = {h->bda_row};
    npy_intp dims2[] = {h->bda_row * h->num_pols};
    ant1   = (PyArrayObject*)PyArray_SimpleNew(1, dims1, NPY_INT);
    ant2   = (PyArrayObject*)PyArray_SimpleNew(1, dims1, NPY_INT);
    uu     = (PyArrayObject*)PyArray_SimpleNew(1, dims1, NPY_DOUBLE);
    vv     = (PyArrayObject*)PyArray_SimpleNew(1, dims1, NPY_DOUBLE);
    ww     = (PyArrayObject*)PyArray_SimpleNew(1, dims1, NPY_DOUBLE);
    data   = (PyArrayObject*)PyArray_SimpleNew(1, dims2, NPY_CDOUBLE);
    expo   = (PyArrayObject*)PyArray_SimpleNew(1, dims1, NPY_DOUBLE);
    sigma  = (PyArrayObject*)PyArray_SimpleNew(1, dims1, NPY_DOUBLE);
    weight = (PyArrayObject*)PyArray_SimpleNew(1, dims1, NPY_DOUBLE);

    /* Copy the data into the arrays. */
    memcpy(PyArray_DATA(ant1), h->ant1, h->bda_row * INT);
    memcpy(PyArray_DATA(ant2), h->ant2, h->bda_row * INT);
    memcpy(PyArray_DATA(uu), h->u, h->bda_row * DBL);
    memcpy(PyArray_DATA(vv), h->v, h->bda_row * DBL);
    memcpy(PyArray_DATA(ww), h->w, h->bda_row * DBL);
    memcpy(PyArray_DATA(data), h->data, h->bda_row * h->num_pols * 2*DBL);
    memcpy(PyArray_DATA(expo), h->exposure, h->bda_row * DBL);
    memcpy(PyArray_DATA(sigma), h->sigma, h->bda_row * DBL);
    memcpy(PyArray_DATA(weight), h->weight, h->bda_row * DBL);

    /* Create a dictionary and return the data in it. */
    dict = PyDict_New();
    PyDict_SetItemString(dict, "antenna1", (PyObject*)ant1);
    PyDict_SetItemString(dict, "antenna2", (PyObject*)ant2);
    PyDict_SetItemString(dict, "uu", (PyObject*)uu);
    PyDict_SetItemString(dict, "vv", (PyObject*)vv);
    PyDict_SetItemString(dict, "ww", (PyObject*)ww);
    PyDict_SetItemString(dict, "data", (PyObject*)data);
    PyDict_SetItemString(dict, "exposure", (PyObject*)expo);
    PyDict_SetItemString(dict, "sigma", (PyObject*)sigma);
    PyDict_SetItemString(dict, "weight", (PyObject*)weight);

    /* Clear all current averages. */
    oskar_bda_clear(h);

    return Py_BuildValue("N", dict); /* Don't increment refcount. */
}


/* Method table. */
static PyMethodDef methods[] =
{
    {
        "apply_gains",
        (PyCFunction)apply_gains, METH_VARARGS,
        "vis_amp = apply_gains(vis_amp, gains)\n"
        "Applies gains.\n"
    },
    {
        "apply_gains_2",
        (PyCFunction)apply_gains_2, METH_VARARGS,
        "vis_amp = apply_gains_2(vis_amp, gains)\n"
        "Applies gains.\n"
    },
    {
        "vis_list_to_matrix",
        (PyCFunction)vis_list_to_matrix, METH_VARARGS,
        "vis_matrix = vis_list_to_matrix(vis_list)\n"
        "Converts a list of visibilities to matrix form.\n"
    },
    {
        "vis_list_to_matrix_2",
        (PyCFunction)vis_list_to_matrix_2, METH_VARARGS,
        "vis_matrix = vis_list_to_matrix_2(vis_list)\n"
        "Converts a list of visibilities to matrix form.\n"
    },
    {
        "check_ref_count",
        (PyCFunction)check_ref_count, METH_VARARGS,
        "count = check_ref_count(PyObject)\n"
        "Check the reference count of a python object\n"
    },
    {
        "expand",
        (PyCFunction)expand, METH_VARARGS,
        "expand(num_antennas)\n"
        "Expands BDA data.\n"
    },
    {
        "bda_create",
        (PyCFunction)bda_create, METH_VARARGS,
        "handle = bda_create(num_antennas, num_pols)\n"
        "Creates the BDA object.\n"
    },
    {
        "bda_set_compression",
        (PyCFunction)bda_set_compression, METH_VARARGS,
        "bda_set_max_fact(max_fact, fov_deg, wavelength_m, max_avg_time_s)\n"
        "Sets compression parameters.\n"
    },
    {
        "bda_set_delta_t",
        (PyCFunction)bda_set_delta_t, METH_VARARGS,
        "bda_set_delta_t(value)\n"
        "Sets time interval of input data.\n"
    },
    {
        "bda_set_num_times",
        (PyCFunction)bda_set_num_times, METH_VARARGS,
        "bda_set_num_times(value)\n"
        "Sets number of times in the input data.\n"
    },
    {
        "bda_set_initial_coords",
        (PyCFunction)bda_set_initial_coords, METH_VARARGS,
        "bda_set_initial_coords(uu, vv, ww)\n"
        "Sets initial baseline coordinates.\n"
    },
    {
        "bda_add_data",
        (PyCFunction)bda_add_data, METH_VARARGS,
        "bda_add_data(time_index, vis, uu_next, vv_next, ww_next)\n"
        "Supplies visibility data to be averaged.\n"
    },
    {
        "bda_finalise",
        (PyCFunction)bda_finalise, METH_VARARGS,
        "averaged_data = bda_finalise()\n"
        "Returns averaged visibility data as a dictionary of arrays.\n"
    },
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_bda_utils",       /* m_name */
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
    m = Py_InitModule3("_bda_utils", methods, "docstring ...");
#endif
    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__bda_utils(void)
{
    import_array();
    return moduleinit();
}
#else
// XXX the init function name has to match that of the compiled module
// with the pattern 'init<module name>'. This module is called '_apply_gains'
void init_bda_utils(void)
{
    import_array();
    moduleinit();
    return;
}
#endif
