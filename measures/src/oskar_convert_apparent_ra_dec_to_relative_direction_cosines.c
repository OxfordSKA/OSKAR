/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#include <oskar_convert_apparent_ra_dec_to_relative_direction_cosines.h>
#include <oskar_convert_apparent_ra_dec_to_relative_direction_cosines_cuda.h>
#include <oskar_convert_apparent_ra_dec_to_relative_direction_cosines_inline.h>
#include <oskar_cuda_check_error.h>

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_convert_apparent_ra_dec_to_relative_direction_cosines_f(int np,
        const float* ra, const float* dec, float ra0, float dec0,
        float* l, float* m, float* n)
{
    int i;
    float sin_dec0, cos_dec0;
    sin_dec0 = (float) sin(dec0);
    cos_dec0 = (float) cos(dec0);

    #pragma omp parallel for private(i)
    for (i = 0; i < np; ++i)
    {
        oskar_convert_apparent_ra_dec_to_relative_direction_cosines_inline_f(
                ra[i], dec[i], ra0, cos_dec0, sin_dec0, &l[i], &m[i], &n[i]);
    }
}

/* Double precision. */
void oskar_convert_apparent_ra_dec_to_relative_direction_cosines_d(int np,
        const double* ra, const double* dec, double ra0, double dec0,
        double* l, double* m, double* n)
{
    int i;
    double sin_dec0, cos_dec0;
    sin_dec0 = sin(dec0);
    cos_dec0 = cos(dec0);

    #pragma omp parallel for private(i)
    for (i = 0; i < np; ++i)
    {
        oskar_convert_apparent_ra_dec_to_relative_direction_cosines_inline_d(
                ra[i], dec[i], ra0, cos_dec0, sin_dec0, &l[i], &m[i], &n[i]);
    }
}

/* Single precision. */
void oskar_convert_apparent_ra_dec_to_relative_direction_cosines_2D_f(int np,
        const float* ra, const float* dec, float ra0, float dec0,
        float* l, float* m)
{
    int i;
    float sinLat0, cosLat0;
    sinLat0 = sinf(dec0);
    cosLat0 = cosf(dec0);

    #pragma omp parallel for private(i)
    for (i = 0; i < np; ++i)
    {
        float cosLat, sinLat, sinLon, cosLon, relLon, pLat, l_, m_;
        pLat = dec[i];
        relLon = ra[i];
        relLon -= ra0;
        sinLon = sinf(relLon);
        cosLon = cosf(relLon);
        sinLat = sinf(pLat);
        cosLat = cosf(pLat);
        l_ = cosLat * sinLon;
        m_ = cosLat0 * sinLat - sinLat0 * cosLat * cosLon;
        l[i] = l_;
        m[i] = m_;
    }
}

/* Double precision. */
void oskar_convert_apparent_ra_dec_to_relative_direction_cosines_2D_d(int np,
        const double* ra, const double* dec, double ra0, double dec0,
        double* l, double* m)
{
    int i;
    double sinLat0, cosLat0;
    sinLat0 = sin(dec0);
    cosLat0 = cos(dec0);

    #pragma omp parallel for private(i)
    for (i = 0; i < np; ++i)
    {
        double cosLat, sinLat, sinLon, cosLon, relLon, pLat, l_, m_;
        pLat = dec[i];
        relLon = ra[i];
        relLon -= ra0;
        sinLon = sin(relLon);
        cosLon = cos(relLon);
        sinLat = sin(pLat);
        cosLat = cos(pLat);
        l_ = cosLat * sinLon;
        m_ = cosLat0 * sinLat - sinLat0 * cosLat * cosLon;
        l[i] = l_;
        m[i] = m_;
    }
}

/* Wrapper. */
void oskar_convert_apparent_ra_dec_to_relative_direction_cosines(int num_points,
        const oskar_Mem* ra, const oskar_Mem* dec, double ra0_rad,
        double dec0_rad, oskar_Mem* l, oskar_Mem* m, oskar_Mem* n, int* status)
{
    int type, location;

    /* Check all inputs. */
    if (!ra || !dec || !l || !m || !n || !status)
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
    if (oskar_mem_type(dec) != type || oskar_mem_type(l) != type ||
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
    if (oskar_mem_location(dec) != location ||
            oskar_mem_location(l) != location ||
            oskar_mem_location(m) != location ||
            oskar_mem_location(n) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check memory is allocated. */
    if (!oskar_mem_allocated(ra) || !oskar_mem_allocated(dec))
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }

    /* Check dimensions. */
    if ((int)oskar_mem_length(ra) < num_points ||
            (int)oskar_mem_length(dec) < num_points)
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
        const float *ra_, *dec_;
        float *l_, *m_, *n_;
        ra_  = oskar_mem_float_const(ra, status);
        dec_ = oskar_mem_float_const(dec, status);
        l_   = oskar_mem_float(l, status);
        m_   = oskar_mem_float(m, status);
        n_   = oskar_mem_float(n, status);

        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_convert_apparent_ra_dec_to_relative_direction_cosines_cuda_f(
                    num_points, ra_, dec_, (float)ra0_rad, (float)dec0_rad,
                    l_, m_, n_);
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else
        {
            oskar_convert_apparent_ra_dec_to_relative_direction_cosines_f(
                    num_points, ra_, dec_, (float)ra0_rad, (float)dec0_rad,
                    l_, m_, n_);
        }
    }
    else
    {
        const double *ra_, *dec_;
        double *l_, *m_, *n_;
        ra_  = oskar_mem_double_const(ra, status);
        dec_ = oskar_mem_double_const(dec, status);
        l_   = oskar_mem_double(l, status);
        m_   = oskar_mem_double(m, status);
        n_   = oskar_mem_double(n, status);

        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_convert_apparent_ra_dec_to_relative_direction_cosines_cuda_d(
                    num_points, ra_, dec_, ra0_rad, dec0_rad, l_, m_, n_);
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else
        {
            oskar_convert_apparent_ra_dec_to_relative_direction_cosines_d(
                    num_points, ra_, dec_, ra0_rad, dec0_rad, l_, m_, n_);
        }
    }
}

#ifdef __cplusplus
}
#endif
