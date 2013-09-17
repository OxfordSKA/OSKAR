/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include <oskar_evaluate_uvw_station.h>
#include <oskar_evaluate_uvw_station_cuda.h>
#include <oskar_cuda_check_error.h>

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_evaluate_uvw_station_f(float* u, float* v, float* w,
        int num_stations, const float* x, const float* y, const float* z,
        double ha0_rad, double dec0_rad)
{
    int i;
    double sinHa0, cosHa0, sinDec0, cosDec0;

    /* Precompute trig. */
    sinHa0  = sin(ha0_rad);
    cosHa0  = cos(ha0_rad);
    sinDec0 = sin(dec0_rad);
    cosDec0 = cos(dec0_rad);

    /* Loop over points. */
    for (i = 0; i < num_stations; ++i)
    {
        double xi, yi, zi, ut, vt, wt;

        /* Get the input coordinates. */
        xi = (double) (x[i]);
        yi = (double) (y[i]);
        zi = (double) (z[i]);

        /* Apply rotation matrix. */
        ut =  xi * sinHa0 + yi * cosHa0;
        vt = sinDec0 * (-xi * cosHa0 + yi * sinHa0) + zi * cosDec0;
        wt = cosDec0 * (xi * cosHa0 - yi * sinHa0) + zi * sinDec0;

        /* Save the rotated values. */
        u[i] = (float)ut;
        v[i] = (float)vt;
        w[i] = (float)wt;
    }
}

/* Double precision. */
void oskar_evaluate_uvw_station_d(double* u, double* v, double* w,
        int num_stations, const double* x, const double* y, const double* z,
        double ha0_rad, double dec0_rad)
{
    int i;
    double sinHa0, cosHa0, sinDec0, cosDec0;

    /* Precompute trig. */
    sinHa0  = sin(ha0_rad);
    cosHa0  = cos(ha0_rad);
    sinDec0 = sin(dec0_rad);
    cosDec0 = cos(dec0_rad);

    /* Loop over points. */
    for (i = 0; i < num_stations; ++i)
    {
        double xi, yi, zi, ut, vt, wt;

        /* Get the input coordinates. */
        xi = x[i];
        yi = y[i];
        zi = z[i];

        /* Apply rotation matrix. */
        ut =  xi * sinHa0 + yi * cosHa0;
        vt = sinDec0 * (-xi * cosHa0 + yi * sinHa0) + zi * cosDec0;
        wt = cosDec0 * (xi * cosHa0 - yi * sinHa0) + zi * sinDec0;

        /* Save the rotated values. */
        u[i] = ut;
        v[i] = vt;
        w[i] = wt;
    }
}

/* Wrapper. */
void oskar_evaluate_uvw_station(oskar_Mem* u, oskar_Mem* v, oskar_Mem* w,
        int num_stations, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, double ra0_rad, double dec0_rad, double gast,
        int* status)
{
    int type, location;
    double ha0_rad;

    /* Check all inputs. */
    if (!u || !v || !w || !x || !y || !z || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get data type and location of the input coordinates. */
    type = oskar_mem_type(x);
    location = oskar_mem_location(x);

    /* Check that the memory is not NULL. */
    if (!u->data || !v->data || !w->data || !x->data || !y->data || !z->data)
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    /* Check that the data dimensions are OK. */
    if ((int)oskar_mem_length(u) < num_stations ||
            (int)oskar_mem_length(v) < num_stations ||
            (int)oskar_mem_length(w) < num_stations ||
            (int)oskar_mem_length(x) < num_stations ||
            (int)oskar_mem_length(y) < num_stations ||
            (int)oskar_mem_length(z) < num_stations)
        *status = OSKAR_ERR_DIMENSION_MISMATCH;

    /* Check that the data are in the right location. */
    if (oskar_mem_location(y) != location ||
            oskar_mem_location(z) != location ||
            oskar_mem_location(u) != location ||
            oskar_mem_location(v) != location ||
            oskar_mem_location(w) != location)
        *status = OSKAR_ERR_BAD_LOCATION;

    /* Check that the data is of the right type. */
    if (oskar_mem_type(y) != type || oskar_mem_type(z) != type ||
            oskar_mem_type(u) != type || oskar_mem_type(v) != type || oskar_mem_type(w) != type)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Evaluate Greenwich Hour Angle of phase centre. */
    ha0_rad = gast - ra0_rad;

    /* Evaluate station u,v,w coordinates. */
    if (location == OSKAR_LOCATION_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_SINGLE)
        {
            oskar_evaluate_uvw_station_cuda_f((float*)(u->data),
                    (float*)(v->data), (float*)(w->data), num_stations,
                    (float*)(x->data), (float*)(y->data), (float*)(z->data),
                    (float)ha0_rad, (float)dec0_rad);
            oskar_cuda_check_error(status);
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_evaluate_uvw_station_cuda_d((double*)(u->data),
                    (double*)(v->data), (double*)(w->data), num_stations,
                    (double*)(x->data), (double*)(y->data), (double*)(z->data),
                    ha0_rad, dec0_rad);
            oskar_cuda_check_error(status);
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location == OSKAR_LOCATION_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            oskar_evaluate_uvw_station_f((float*)(u->data),
                    (float*)(v->data), (float*)(w->data), num_stations,
                    (float*)(x->data), (float*)(y->data), (float*)(z->data),
                    (float)ha0_rad, (float)dec0_rad);
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_evaluate_uvw_station_d((double*)(u->data),
                    (double*)(v->data), (double*)(w->data), num_stations,
                    (double*)(x->data), (double*)(y->data), (double*)(z->data),
                    ha0_rad, dec0_rad);
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
}

#ifdef __cplusplus
}
#endif
