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

#include "convert/oskar_convert_lon_lat_to_relative_directions.h"
#include "convert/oskar_convert_lon_lat_to_relative_directions_cuda.h"
#include "convert/private_convert_lon_lat_to_relative_directions_inline.h"
#include "utility/oskar_device_utils.h"

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_convert_lon_lat_to_relative_directions_f(int num_points,
        const float* lon_rad, const float* lat_rad, float lon0_rad,
        float lat0_rad, float* l, float* m, float* n)
{
    int i;
    float sin_lat0, cos_lat0;
    sin_lat0 = (float) sin(lat0_rad);
    cos_lat0 = (float) cos(lat0_rad);

    #pragma omp parallel for private(i)
    for (i = 0; i < num_points; ++i)
    {
        oskar_convert_lon_lat_to_relative_directions_inline_f(
                lon_rad[i], lat_rad[i], lon0_rad, cos_lat0, sin_lat0,
                &l[i], &m[i], &n[i]);
    }
}

/* Double precision. */
void oskar_convert_lon_lat_to_relative_directions_d(int num_points,
        const double* lon_rad, const double* lat_rad, double lon0_rad,
        double lat0_rad, double* l, double* m, double* n)
{
    int i;
    double sin_lat0, cos_lat0;
    sin_lat0 = sin(lat0_rad);
    cos_lat0 = cos(lat0_rad);

    #pragma omp parallel for private(i)
    for (i = 0; i < num_points; ++i)
    {
        oskar_convert_lon_lat_to_relative_directions_inline_d(
                lon_rad[i], lat_rad[i], lon0_rad, cos_lat0, sin_lat0,
                &l[i], &m[i], &n[i]);
    }
}

/* Single precision. */
void oskar_convert_lon_lat_to_relative_directions_2d_f(int num_points,
        const float* lon_rad, const float* lat_rad, float lon0_rad,
        float lat0_rad, float* l, float* m)
{
    int i;
    float sin_lat0, cos_lat0;
    sin_lat0 = sinf(lat0_rad);
    cos_lat0 = cosf(lat0_rad);

    #pragma omp parallel for private(i)
    for (i = 0; i < num_points; ++i)
    {
        float cos_lat, sin_lat, sin_lon, cos_lon, rel_lon, p_lat, l_, m_;
        p_lat = lat_rad[i];
        rel_lon = lon_rad[i];
        rel_lon -= lon0_rad;
        sin_lon = sinf(rel_lon);
        cos_lon = cosf(rel_lon);
        sin_lat = sinf(p_lat);
        cos_lat = cosf(p_lat);
        l_ = cos_lat * sin_lon;
        m_ = cos_lat0 * sin_lat - sin_lat0 * cos_lat * cos_lon;
        l[i] = l_;
        m[i] = m_;
    }
}

/* Double precision. */
void oskar_convert_lon_lat_to_relative_directions_2d_d(int num_points,
        const double* lon_rad, const double* lat_rad, double lon0_rad,
        double lat0_rad, double* l, double* m)
{
    int i;
    double sin_lat0, cos_lat0;
    sin_lat0 = sin(lat0_rad);
    cos_lat0 = cos(lat0_rad);

    #pragma omp parallel for private(i)
    for (i = 0; i < num_points; ++i)
    {
        double cos_lat, sin_lat, sin_lon, cos_lon, rel_lon, p_lat, l_, m_;
        p_lat = lat_rad[i];
        rel_lon = lon_rad[i];
        rel_lon -= lon0_rad;
        sin_lon = sin(rel_lon);
        cos_lon = cos(rel_lon);
        sin_lat = sin(p_lat);
        cos_lat = cos(p_lat);
        l_ = cos_lat * sin_lon;
        m_ = cos_lat0 * sin_lat - sin_lat0 * cos_lat * cos_lon;
        l[i] = l_;
        m[i] = m_;
    }
}

/* Wrapper. */
void oskar_convert_lon_lat_to_relative_directions(int num_points,
        const oskar_Mem* lon_rad, const oskar_Mem* lat_rad, double lon0_rad,
        double lat0_rad, oskar_Mem* l, oskar_Mem* m, oskar_Mem* n, int* status)
{
    int type, location;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the meta-data. */
    type = oskar_mem_type(lon_rad);
    location = oskar_mem_location(lon_rad);

    /* Check type consistency. */
    if (oskar_mem_type(lat_rad) != type || oskar_mem_type(l) != type ||
            oskar_mem_type(m) != type || oskar_mem_type(n) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Check location consistency. */
    if (oskar_mem_location(lat_rad) != location ||
            oskar_mem_location(l) != location ||
            oskar_mem_location(m) != location ||
            oskar_mem_location(n) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check memory is allocated. */
    if (!oskar_mem_allocated(lon_rad) || !oskar_mem_allocated(lat_rad))
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }

    /* Check dimensions. */
    if ((int)oskar_mem_length(lon_rad) < num_points ||
            (int)oskar_mem_length(lat_rad) < num_points)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Resize output arrays if needed. */
    if ((int)oskar_mem_length(l) < num_points)
        oskar_mem_realloc(l, num_points, status);
    if ((int)oskar_mem_length(m) < num_points)
        oskar_mem_realloc(m, num_points, status);
    if ((int)oskar_mem_length(n) < num_points)
        oskar_mem_realloc(n, num_points, status);

    /* Check if safe to proceed. */
    if (*status) return;

    /* Convert coordinates. */
    if (type == OSKAR_SINGLE)
    {
        const float *lon_, *lat_;
        float *l_, *m_, *n_;
        lon_ = oskar_mem_float_const(lon_rad, status);
        lat_ = oskar_mem_float_const(lat_rad, status);
        l_   = oskar_mem_float(l, status);
        m_   = oskar_mem_float(m, status);
        n_   = oskar_mem_float(n, status);

        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_convert_lon_lat_to_relative_directions_cuda_f(
                    num_points, lon_, lat_, (float)lon0_rad, (float)lat0_rad,
                    l_, m_, n_);
            oskar_device_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else
        {
            oskar_convert_lon_lat_to_relative_directions_f(
                    num_points, lon_, lat_, (float)lon0_rad, (float)lat0_rad,
                    l_, m_, n_);
        }
    }
    else
    {
        const double *lon_, *lat_;
        double *l_, *m_, *n_;
        lon_ = oskar_mem_double_const(lon_rad, status);
        lat_ = oskar_mem_double_const(lat_rad, status);
        l_   = oskar_mem_double(l, status);
        m_   = oskar_mem_double(m, status);
        n_   = oskar_mem_double(n, status);

        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_convert_lon_lat_to_relative_directions_cuda_d(
                    num_points, lon_, lat_, lon0_rad, lat0_rad, l_, m_, n_);
            oskar_device_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else
        {
            oskar_convert_lon_lat_to_relative_directions_d(
                    num_points, lon_, lat_, lon0_rad, lat0_rad, l_, m_, n_);
        }
    }
}

#ifdef __cplusplus
}
#endif
