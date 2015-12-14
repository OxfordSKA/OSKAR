/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#include <oskar_convert_ecef_to_station_uvw.h>
#include <oskar_convert_ecef_to_station_uvw_cuda.h>
#include <oskar_cuda_check_error.h>
#include <private_convert_ecef_to_station_uvw_inline.h>

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_convert_ecef_to_station_uvw_f(int num_stations, const float* x,
        const float* y, const float* z, double ha0_rad, double dec0_rad,
        float* u, float* v, float* w)
{
    int i;
    double sin_ha0, cos_ha0, sin_dec0, cos_dec0;

    /* Precompute trig. */
    sin_ha0  = sin(ha0_rad);
    cos_ha0  = cos(ha0_rad);
    sin_dec0 = sin(dec0_rad);
    cos_dec0 = cos(dec0_rad);

    /* Loop over points. */
    for (i = 0; i < num_stations; ++i)
    {
        double ut, vt, wt;
        oskar_convert_ecef_to_station_uvw_inline_d(x[i], y[i], z[i],
                sin_ha0, cos_ha0, sin_dec0, cos_dec0, &ut, &vt, &wt);
        u[i] = (float)ut;
        v[i] = (float)vt;
        w[i] = (float)wt;
    }
}

/* Double precision. */
void oskar_convert_ecef_to_station_uvw_d(int num_stations, const double* x,
        const double* y, const double* z, double ha0_rad, double dec0_rad,
        double* u, double* v, double* w)
{
    int i;
    double sin_ha0, cos_ha0, sin_dec0, cos_dec0;

    /* Precompute trig. */
    sin_ha0  = sin(ha0_rad);
    cos_ha0  = cos(ha0_rad);
    sin_dec0 = sin(dec0_rad);
    cos_dec0 = cos(dec0_rad);

    /* Loop over points. */
    for (i = 0; i < num_stations; ++i)
    {
        oskar_convert_ecef_to_station_uvw_inline_d(x[i], y[i], z[i], sin_ha0,
                cos_ha0, sin_dec0, cos_dec0, &u[i], &v[i], &w[i]);
    }
}

/* Wrapper. */
void oskar_convert_ecef_to_station_uvw(int num_stations, const oskar_Mem* x,
        const oskar_Mem* y, const oskar_Mem* z, double ra0_rad,
        double dec0_rad, double gast, oskar_Mem* u, oskar_Mem* v,
        oskar_Mem* w, int* status)
{
    int type, location;
    double ha0_rad;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get data type and location of the input coordinates. */
    type = oskar_mem_type(x);
    location = oskar_mem_location(x);

    /* Check that the memory is allocated. */
    if (!oskar_mem_allocated(u) || !oskar_mem_allocated(v) ||
            !oskar_mem_allocated(w) || !oskar_mem_allocated(x) ||
            !oskar_mem_allocated(y) || !oskar_mem_allocated(z))
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }

    /* Check that the data dimensions are OK. */
    if ((int)oskar_mem_length(u) < num_stations ||
            (int)oskar_mem_length(v) < num_stations ||
            (int)oskar_mem_length(w) < num_stations ||
            (int)oskar_mem_length(x) < num_stations ||
            (int)oskar_mem_length(y) < num_stations ||
            (int)oskar_mem_length(z) < num_stations)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Check that the data are in the right location. */
    if (oskar_mem_location(y) != location ||
            oskar_mem_location(z) != location ||
            oskar_mem_location(u) != location ||
            oskar_mem_location(v) != location ||
            oskar_mem_location(w) != location)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Check that the data is of the right type. */
    if (oskar_mem_type(y) != type || oskar_mem_type(z) != type ||
            oskar_mem_type(u) != type || oskar_mem_type(v) != type ||
            oskar_mem_type(w) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Evaluate Greenwich Hour Angle of phase centre. */
    ha0_rad = gast - ra0_rad;

    /* Evaluate station u,v,w coordinates. */
    if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_SINGLE)
        {
            oskar_convert_ecef_to_station_uvw_cuda_f(num_stations,
                    oskar_mem_float_const(x, status),
                    oskar_mem_float_const(y, status),
                    oskar_mem_float_const(z, status),
                    (float)ha0_rad, (float)dec0_rad,
                    oskar_mem_float(u, status),
                    oskar_mem_float(v, status),
                    oskar_mem_float(w, status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_convert_ecef_to_station_uvw_cuda_d(num_stations,
                    oskar_mem_double_const(x, status),
                    oskar_mem_double_const(y, status),
                    oskar_mem_double_const(z, status),
                    ha0_rad, dec0_rad,
                    oskar_mem_double(u, status),
                    oskar_mem_double(v, status),
                    oskar_mem_double(w, status));
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
        oskar_cuda_check_error(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            oskar_convert_ecef_to_station_uvw_f(num_stations,
                    oskar_mem_float_const(x, status),
                    oskar_mem_float_const(y, status),
                    oskar_mem_float_const(z, status),
                    (float)ha0_rad, (float)dec0_rad,
                    oskar_mem_float(u, status),
                    oskar_mem_float(v, status),
                    oskar_mem_float(w, status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_convert_ecef_to_station_uvw_d(num_stations,
                    oskar_mem_double_const(x, status),
                    oskar_mem_double_const(y, status),
                    oskar_mem_double_const(z, status),
                    ha0_rad, dec0_rad,
                    oskar_mem_double(u, status),
                    oskar_mem_double(v, status),
                    oskar_mem_double(w, status));
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
