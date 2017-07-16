/*
 * Copyright (c) 2014-2015, The University of Oxford
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

#include "convert/oskar_convert_cirs_relative_directions_to_enu_directions.h"
#include "convert/oskar_convert_cirs_relative_directions_to_enu_directions_cuda.h"
#include "convert/private_convert_cirs_relative_directions_to_enu_directions_inline.h"
#include "convert/private_evaluate_cirs_observed_parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_convert_cirs_relative_directions_to_enu_directions_f(
        int num_points, const float* l, const float* m, const float* n,
        float ra0_rad, float dec0_rad, float lon_rad, float lat_rad,
        float era_rad, float pm_x_rad, float pm_y_rad,
        float diurnal_aberration, float* x, float* y, float* z)
{
    int i;
    double sin_lat, cos_lat, sin_ha0, cos_ha0, sin_dec0, cos_dec0;
    double local_pm_x, local_pm_y;

    /* Calculate common transform parameters. */
    oskar_evaluate_cirs_observed_parameters(lon_rad, lat_rad, era_rad,
            ra0_rad, dec0_rad, pm_x_rad, pm_y_rad, &sin_lat, &cos_lat, &sin_ha0,
            &cos_ha0, &sin_dec0, &cos_dec0, &local_pm_x, &local_pm_y);

    /* Loop over positions. */
    for (i = 0; i < num_points; ++i)
    {
        oskar_convert_cirs_relative_directions_to_enu_directions_inline_f(
                l[i], m[i], n[i], sin_lat, cos_lat, sin_ha0, cos_ha0,
                sin_dec0, cos_dec0, local_pm_x, local_pm_y,
                diurnal_aberration, &x[i], &y[i], &z[i]);
    }
}

/* Double precision. */
void oskar_convert_cirs_relative_directions_to_enu_directions_d(
        int num_points, const double* l, const double* m, const double* n,
        double ra0_rad, double dec0_rad, double lon_rad, double lat_rad,
        double era_rad, double pm_x_rad, double pm_y_rad,
        double diurnal_aberration, double* x, double* y, double* z)
{
    int i;
    double sin_lat, cos_lat, sin_ha0, cos_ha0, sin_dec0, cos_dec0;
    double local_pm_x, local_pm_y;

    /* Calculate common transform parameters. */
    oskar_evaluate_cirs_observed_parameters(lon_rad, lat_rad, era_rad,
            ra0_rad, dec0_rad, pm_x_rad, pm_y_rad, &sin_lat, &cos_lat, &sin_ha0,
            &cos_ha0, &sin_dec0, &cos_dec0, &local_pm_x, &local_pm_y);

    /* Loop over positions. */
    for (i = 0; i < num_points; ++i)
    {
        oskar_convert_cirs_relative_directions_to_enu_directions_inline_d(
                l[i], m[i], n[i], sin_lat, cos_lat, sin_ha0, cos_ha0,
                sin_dec0, cos_dec0, local_pm_x, local_pm_y,
                diurnal_aberration, &x[i], &y[i], &z[i]);
    }
}

/* Wrapper. */
void oskar_convert_cirs_relative_directions_to_enu_directions(int num_points,
        const oskar_Mem* l, const oskar_Mem* m, const oskar_Mem* n,
        double ra0_rad, double dec0_rad, double lon_rad, double lat_rad,
        double era_rad, double pm_x_rad, double pm_y_rad,
        double diurnal_aberration, oskar_Mem* x, oskar_Mem* y, oskar_Mem* z,
        int* status)
{
    int type, location;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get type and check consistency. */
    type = oskar_mem_type(x);
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (type != oskar_mem_type(y) || type != oskar_mem_type(z) ||
            type != oskar_mem_type(l) || type != oskar_mem_type(m) ||
            type != oskar_mem_type(n))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Get location and check consistency. */
    location = oskar_mem_location(x);
    if (location != oskar_mem_location(y) ||
            location != oskar_mem_location(z) ||
            location != oskar_mem_location(l) ||
            location != oskar_mem_location(m) ||
            location != oskar_mem_location(n))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check dimension consistency. */
    if ((int)oskar_mem_length(x) < num_points ||
            (int)oskar_mem_length(y) < num_points ||
            (int)oskar_mem_length(z) < num_points ||
            (int)oskar_mem_length(l) < num_points ||
            (int)oskar_mem_length(m) < num_points ||
            (int)oskar_mem_length(n) < num_points)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Switch on type and location. */
    if (type == OSKAR_DOUBLE)
    {
        double *x_, *y_, *z_;
        const double *l_, *m_, *n_;
        x_ = oskar_mem_double(x, status);
        y_ = oskar_mem_double(y, status);
        z_ = oskar_mem_double(z, status);
        l_ = oskar_mem_double_const(l, status);
        m_ = oskar_mem_double_const(m, status);
        n_ = oskar_mem_double_const(n, status);

        if (location == OSKAR_CPU)
        {
            oskar_convert_cirs_relative_directions_to_enu_directions_d(
                    num_points, l_, m_, n_, ra0_rad, dec0_rad, lon_rad,
                    lat_rad, era_rad, pm_x_rad, pm_y_rad, diurnal_aberration,
                    x_, y_, z_);
        }
        else if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_convert_cirs_relative_directions_to_enu_directions_cuda_d(
                    num_points, l_, m_, n_, ra0_rad, dec0_rad, lon_rad,
                    lat_rad, era_rad, pm_x_rad, pm_y_rad, diurnal_aberration,
                    x_, y_, z_);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
    }
    else
    {
        float *x_, *y_, *z_;
        const float *l_, *m_, *n_;
        x_ = oskar_mem_float(x, status);
        y_ = oskar_mem_float(y, status);
        z_ = oskar_mem_float(z, status);
        l_ = oskar_mem_float_const(l, status);
        m_ = oskar_mem_float_const(m, status);
        n_ = oskar_mem_float_const(n, status);

        if (location == OSKAR_CPU)
        {
            oskar_convert_cirs_relative_directions_to_enu_directions_f(
                    num_points, l_, m_, n_, ra0_rad, dec0_rad, lon_rad,
                    lat_rad, era_rad, pm_x_rad, pm_y_rad, diurnal_aberration,
                    x_, y_, z_);
        }
        else if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_convert_cirs_relative_directions_to_enu_directions_cuda_f(
                    num_points, l_, m_, n_, ra0_rad, dec0_rad, lon_rad,
                    lat_rad, era_rad, pm_x_rad, pm_y_rad, diurnal_aberration,
                    x_, y_, z_);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
    }
}

#ifdef __cplusplus
}
#endif
