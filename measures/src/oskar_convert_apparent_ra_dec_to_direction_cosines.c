/*
 * Copyright (c) 2013, The University of Oxford
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

#include <oskar_convert_apparent_ra_dec_to_direction_cosines.h>

#include <oskar_mem.h>
#include <utility/oskar_cuda_check_error.h>
#include <oskar_convert_apparent_ra_dec_to_direction_cosines_cuda.h>

#include <math.h>

/*#include "oskar_compute_tangent_plane_direction_z.h"*/
/*#include "oskar_convert_lon_lat_to_tangent_plane_direction.h"*/

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_convert_apparent_ra_dec_to_direction_cosines_f(int n,
        const float* ra, const float* dec, float ra0, float dec0,
        float* x, float* y, float* z)
{
    /* Compute x,y-direction-cosines of RA, Dec relative to reference point. */
    /*oskar_convert_lon_lat_to_tangent_plane_direction_omp_f(num_points,
            ra0_rad, dec0_rad, ra, dec, x, y); */
    /* Compute z-direction-cosines of points from x and y. */
    /*oskar_compute_tangent_plane_direction_z_f(num_points, x, y, z);*/

    int i;
    float sinLat0, cosLat0;
    sinLat0 = sinf(dec0);
    cosLat0 = cosf(dec0);

    #pragma omp parallel for
    for (i = 0; i < n; ++i)
    {
        float cosLat, sinLat, sinLon, cosLon, relLon, pLat, x_, y_, a;
        pLat = dec[i];
        relLon = ra[i];
        relLon -= ra0;
        sinLon = sinf(relLon);
        cosLon = cosf(relLon);
        sinLat = sinf(pLat);
        cosLat = cosf(pLat);
        x_ = cosLat * sinLon;
        y_ = cosLat0 * sinLat - sinLat0 * cosLat * cosLon;
        x[i] = x_;
        y[i] = y_;
        a = 1.0f - x_*x_ * y_*y_;
        /* FIXME this is z-1 update this function AND use in calling functions. */
        if (a < 0.0f)
            z[i] = -1.0f;
        else
            z[i] = sqrtf(a) - 1.0f;
    }
}

/* Double precision. */
void oskar_convert_apparent_ra_dec_to_direction_cosines_d(int n,
        const double* ra, const double* dec, double ra0, double dec0,
        double* x, double* y, double* z)
{

    /* Compute l,m-direction-cosines of RA, Dec relative to reference point. */
    /*oskar_convert_lon_lat_to_tangent_plane_direction_omp_d(num_points,
            ra0_rad, dec0_rad, ra, dec, x, y); */

    /* Compute z-direction-cosines of points from x and y. */
    /*oskar_compute_tangent_plane_direction_z_d(num_points, x, y, z); */

    int i;
    double sinLat0, cosLat0;
    sinLat0 = sin(dec0);
    cosLat0 = cos(dec0);

    #pragma omp parallel for
    for (i = 0; i < n; ++i)
    {
        double cosLat, sinLat, sinLon, cosLon, relLon, pLat, x_, y_, a;
        pLat = dec[i];
        relLon = ra[i];
        relLon -= ra0;
        sinLon = sin(relLon);
        cosLon = cos(relLon);
        sinLat = sin(pLat);
        cosLat = cos(pLat);
        x_ = cosLat * sinLon;
        y_ = cosLat0 * sinLat - sinLat0 * cosLat * cosLon;
        x[i] = x_;
        y[i] = y_;
        a = 1.0f - x_*x_ * y_*y_;
        /* FIXME this is z-1 update this function AND use in calling functions. */
        if (a < 0.0f)
            z[i] = -1.0f;
        else
            z[i] = sqrt(a) - 1.0f;
    }
}

/* Wrapper. */
void oskar_convert_apparent_ra_dec_to_direction_cosines(int num_points,
        const oskar_Mem* ra, const oskar_Mem* dec, double ra0_rad,
        double dec0_rad, oskar_Mem* x, oskar_Mem* y, oskar_Mem* z,  int* status)
{
    int type, location;

    /* Check all inputs. */
    if (!ra || !dec || !x || !y || !z || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the meta-data. */
    type = oskar_mem_type(ra);
    location = oskar_mem_location(ra);

    /* Check type consistency. */
    if (oskar_mem_type(dec) != type || oskar_mem_type(x) != type ||
            oskar_mem_type(y) != type || oskar_mem_type(z) != type)
        *status = OSKAR_ERR_TYPE_MISMATCH;
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
        *status = OSKAR_ERR_BAD_DATA_TYPE;

    /* Check location consistency. */
    if (oskar_mem_location(dec) != location ||
            oskar_mem_location(x) != location ||
            oskar_mem_location(y) != location ||
            oskar_mem_location(z) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
    }

    /* Check memory is allocated. */
    if (!ra->data || !dec->data)
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    /* Check dimensions. */
    if ((int)oskar_mem_length(ra) < num_points
            || (int)oskar_mem_length(dec) < num_points)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
    }

    /* Resize output arrays if needed. */
    if ((int)oskar_mem_length(x) < num_points)
        oskar_mem_realloc(x, num_points, status);
    if ((int)oskar_mem_length(y) < num_points)
        oskar_mem_realloc(y, num_points, status);
    if ((int)oskar_mem_length(z) < num_points)
        oskar_mem_realloc(z, num_points, status);

    /* Check if safe to proceed. */
    if (*status) return;

    /* Convert coordinates. */
    if (location == OSKAR_LOCATION_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_SINGLE)
        {
            oskar_convert_apparent_ra_dec_to_direction_cosines_cuda_f(
                    num_points, (const float*)ra->data, (const float*)dec->data,
                    (float)ra0_rad, (float)dec0_rad, (float*)x->data,
                    (float*)y->data, (float*)z->data);
            oskar_cuda_check_error(status);
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_convert_apparent_ra_dec_to_direction_cosines_cuda_d(
                    num_points, (const double*)ra->data, (const double*)dec->data,
                    ra0_rad, dec0_rad, (double*)x->data, (double*)y->data,
                    (double*)z->data);
            oskar_cuda_check_error(status);
        }
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location == OSKAR_LOCATION_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            oskar_convert_apparent_ra_dec_to_direction_cosines_f(
                    num_points, (const float*)ra->data, (const float*)dec->data,
                    (float)ra0_rad, (float)dec0_rad, (float*)x->data,
                    (float*)y->data, (float*)z->data);
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_convert_apparent_ra_dec_to_direction_cosines_d(
                    num_points, (const double*)ra->data,
                    (const double*)dec->data, ra0_rad, dec0_rad,
                    (double*)x->data, (double*)y->data, (double*)z->data);
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
