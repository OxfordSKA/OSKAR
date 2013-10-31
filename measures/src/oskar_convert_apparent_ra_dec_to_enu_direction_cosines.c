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

#include <oskar_convert_apparent_ra_dec_to_enu_direction_cosines.h>
#include <oskar_convert_apparent_ha_dec_to_enu_direction_cosines.h>
#include <oskar_convert_apparent_ra_dec_to_enu_direction_cosines_cuda.h>
#include <oskar_cuda_check_error.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_convert_apparent_ra_dec_to_enu_direction_cosines_f(int n,
        const float* ra, const float* dec, float lst, float lat, float* x,
        float* y, float* z)
{
    /* Determine source Hour Angles (HA = LST - RA). */
    float* ha = z; /* Temporary. */
    int i;
    for (i = 0; i < n; ++i)
    {
        ha[i] = lst - ra[i];
    }

    /* Determine horizontal x,y,z directions (destroys contents of ha). */
    oskar_convert_apparent_ha_dec_to_enu_direction_cosines_f(n, ha, dec, lat, x, y, z);
}

/* Double precision. */
void oskar_convert_apparent_ra_dec_to_enu_direction_cosines_d(int n,
        const double* ra, const double* dec, double lst, double lat,
        double* x, double* y, double* z)
{
    /* Determine source Hour Angles (HA = LST - RA). */
    double* ha = z; /* Temporary. */
    int i;
    for (i = 0; i < n; ++i)
    {
        ha[i] = lst - ra[i];
    }

    /* Determine horizontal x,y,z directions (destroys contents of ha). */
    oskar_convert_apparent_ha_dec_to_enu_direction_cosines_d(n, ha, dec, lat, x, y, z);
}

/* oskar_Mem wrapper */
OSKAR_EXPORT
void oskar_convert_apparent_ra_dec_to_enu_direction_cosines(int n, oskar_Mem* x,
        oskar_Mem* y, oskar_Mem* z, const oskar_Mem* ra, const oskar_Mem* dec,
        double last, double lat, int* status)
{
    int type = 0, location;

    /* Check all inputs. */
    if (!ra || !dec || !x || !y || !z || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check that the dimensions are correct. */
    if (n > (int)oskar_mem_length(ra) ||
            n > (int)oskar_mem_length(dec))
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Check type. */
    type = oskar_mem_type(ra);
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (type != oskar_mem_type(dec) || type != oskar_mem_type(x) ||
            type != oskar_mem_type(y) || type != oskar_mem_type(z))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Check location. */
    location = oskar_mem_location(ra);
    if (location != OSKAR_LOCATION_CPU && location != OSKAR_LOCATION_GPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }
    if (location != oskar_mem_location(dec) ||
            location != oskar_mem_location(x) ||
            location != oskar_mem_location(y) ||
            location != oskar_mem_location(z))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Resize output arrays if needed. */
    if ((int)oskar_mem_length(x) < n)
        oskar_mem_realloc(x, n, status);
    if ((int)oskar_mem_length(y) < n)
        oskar_mem_realloc(y, n, status);
    if ((int)oskar_mem_length(z) < n)
        oskar_mem_realloc(z, n, status);
    if (*status) return;

    /* Switch on type and location. */
    if (type == OSKAR_DOUBLE)
    {
        const double *ra_, *dec_;
        double *x_, *y_, *z_;
        ra_ = oskar_mem_double_const(ra, status);
        dec_ = oskar_mem_double_const(dec, status);
        x_ = oskar_mem_double(x, status);
        y_ = oskar_mem_double(y, status);
        z_ = oskar_mem_double(z, status);

        if (location == OSKAR_LOCATION_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_convert_apparent_ra_dec_to_enu_direction_cosines_cuda_d(
                    n, ra_, dec_, last, lat, x_, y_, z_);
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else
        {
            oskar_convert_apparent_ra_dec_to_enu_direction_cosines_d(n,
                    ra_, dec_, last, lat, x_, y_, z_);
        }
    }
    else
    {
        const float *ra_, *dec_;
        float *x_, *y_, *z_;
        ra_ = oskar_mem_float_const(ra, status);
        dec_ = oskar_mem_float_const(dec, status);
        x_ = oskar_mem_float(x, status);
        y_ = oskar_mem_float(y, status);
        z_ = oskar_mem_float(z, status);

        if (location == OSKAR_LOCATION_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_convert_apparent_ra_dec_to_enu_direction_cosines_cuda_f(
                    n, ra_, dec_, (float)last, (float)lat,
                    x_, y_, z_);
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else
        {
            oskar_convert_apparent_ra_dec_to_enu_direction_cosines_f(n,
                    ra_, dec_, (float)last, (float)lat, x_, y_, z_);
        }
    }
}

#ifdef __cplusplus
}
#endif
