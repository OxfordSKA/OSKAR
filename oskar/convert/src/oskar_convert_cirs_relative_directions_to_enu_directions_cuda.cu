/*
 * Copyright (c) 2014, The University of Oxford
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

#include "convert/oskar_convert_cirs_relative_directions_to_enu_directions_cuda.h"
#include "convert/private_convert_cirs_relative_directions_to_enu_directions_inline.h"
#include "convert/private_evaluate_cirs_observed_parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_convert_cirs_relative_directions_to_enu_directions_cuda_f(
        int num_points, const float* d_l, const float* d_m, const float* d_n,
        float ra0_rad, float dec0_rad, float lon_rad, float lat_rad,
        float era_rad, float pm_x_rad, float pm_y_rad,
        float diurnal_aberration, float* d_x, float* d_y, float* d_z)
{
    double sin_lat, cos_lat, sin_ha0, cos_ha0, sin_dec0, cos_dec0;
    double local_pm_x, local_pm_y;
    int num_blocks, num_threads = 256;

    /* Set up thread blocks. */
    num_blocks = (num_points + num_threads - 1) / num_threads;

    /* Calculate common transform parameters. */
    oskar_evaluate_cirs_observed_parameters(lon_rad, lat_rad, era_rad,
            ra0_rad, dec0_rad, pm_x_rad, pm_y_rad, &sin_lat, &cos_lat, &sin_ha0,
            &cos_ha0, &sin_dec0, &cos_dec0, &local_pm_x, &local_pm_y);

    /* Call kernel. */
    oskar_convert_cirs_relative_directions_to_enu_directions_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_points, d_l, d_m, d_n,
            (float)sin_lat, (float)cos_lat, (float)sin_ha0, (float)cos_ha0,
            (float)sin_dec0, (float)cos_dec0, (float)local_pm_x,
            (float)local_pm_y, diurnal_aberration, d_x, d_y, d_z);
}

/* Double precision. */
void oskar_convert_cirs_relative_directions_to_enu_directions_cuda_d(
        int num_points, const double* d_l, const double* d_m, const double* d_n,
        double ra0_rad, double dec0_rad, double lon_rad, double lat_rad,
        double era_rad, double pm_x_rad, double pm_y_rad,
        double diurnal_aberration, double* d_x, double* d_y, double* d_z)
{
    double sin_lat, cos_lat, sin_ha0, cos_ha0, sin_dec0, cos_dec0;
    double local_pm_x, local_pm_y;
    int num_blocks, num_threads = 256;

    /* Set up thread blocks. */
    num_blocks = (num_points + num_threads - 1) / num_threads;

    /* Calculate common transform parameters. */
    oskar_evaluate_cirs_observed_parameters(lon_rad, lat_rad, era_rad,
            ra0_rad, dec0_rad, pm_x_rad, pm_y_rad, &sin_lat, &cos_lat, &sin_ha0,
            &cos_ha0, &sin_dec0, &cos_dec0, &local_pm_x, &local_pm_y);

    /* Call kernel. */
    oskar_convert_cirs_relative_directions_to_enu_directions_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_points, d_l, d_m, d_n,
            sin_lat, cos_lat, sin_ha0, cos_ha0, sin_dec0, cos_dec0,
            local_pm_x, local_pm_y, diurnal_aberration, d_x, d_y, d_z);
}

#ifdef __cplusplus
}
#endif


/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_convert_cirs_relative_directions_to_enu_directions_cudak_f(
        const int num_points, const float* restrict l,
        const float* restrict m, const float* restrict n,
        const float sin_lat, const float cos_lat, const float sin_ha0,
        const float cos_ha0, const float sin_dec0, const float cos_dec0,
        const float local_pm_x, const float local_pm_y,
        const float diurnal_aberration, float* restrict x,
        float* restrict y, float* restrict z)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_points) return;

    oskar_convert_cirs_relative_directions_to_enu_directions_inline_f(
            l[i], m[i], n[i], sin_lat, cos_lat, sin_ha0, cos_ha0, sin_dec0,
            cos_dec0, local_pm_x, local_pm_y, diurnal_aberration,
            &x[i], &y[i], &z[i]);
}

/* Double precision. */
__global__
void oskar_convert_cirs_relative_directions_to_enu_directions_cudak_d(
        const int num_points, const double* restrict l,
        const double* restrict m, const double* restrict n,
        const double sin_lat, const double cos_lat, const double sin_ha0,
        const double cos_ha0, const double sin_dec0, const double cos_dec0,
        const double local_pm_x, const double local_pm_y,
        const double diurnal_aberration, double* restrict x,
        double* restrict y, double* restrict z)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_points) return;

    oskar_convert_cirs_relative_directions_to_enu_directions_inline_d(
            l[i], m[i], n[i], sin_lat, cos_lat, sin_ha0, cos_ha0, sin_dec0,
            cos_dec0, local_pm_x, local_pm_y, diurnal_aberration,
            &x[i], &y[i], &z[i]);
}

