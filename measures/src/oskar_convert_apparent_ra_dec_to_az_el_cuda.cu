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

#include "oskar_convert_apparent_ra_dec_to_az_el_cuda.h"

#include "oskar_convert_apparent_ra_dec_to_enu_direction_cosines_cuda.h"
#include "oskar_convert_enu_direction_cosines_to_az_el_cuda.h"

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Single precision.
int oskar_convert_apparent_ra_dec_to_az_el_cuda_f(int n, const float* d_ra,
        const float* d_dec, float lst, float lat, float* d_work,
        float* d_az, float* d_el)
{
    // Determine horizontal coordinates.
    const int n_thd = 128;
    const int n_blk = (n + n_thd - 1) / n_thd;
    float cosLat = cosf(lat);
    float sinLat = sinf(lat);
    oskar_convert_apparent_ra_dec_to_enu_direction_cosines_cudak_f
    OSKAR_CUDAK_CONF(n_blk, n_thd) (n, d_ra, d_dec, cosLat, sinLat, lst,
            d_az, d_el, d_work);
    oskar_convert_enu_direction_cosines_to_az_el_cudak_f
    OSKAR_CUDAK_CONF(n_blk, n_thd) (n, d_az, d_el, d_work, d_az, d_el);
    cudaDeviceSynchronize();
    return (int)cudaPeekAtLastError();
}

// Double precision.
int oskar_convert_apparent_ra_dec_to_az_el_cuda_d(int n, const double* d_ra,
        const double* d_dec, double lst, double lat, double* d_work,
        double* d_az, double* d_el)
{
    // Determine horizontal coordinates.
    const int n_thd = 128;
    const int n_blk = (n + n_thd - 1) / n_thd;
    double cosLat = cos(lat);
    double sinLat = sin(lat);
    oskar_convert_apparent_ra_dec_to_enu_direction_cosines_cudak_d
    OSKAR_CUDAK_CONF(n_blk, n_thd) (n, d_ra, d_dec, cosLat, sinLat, lst,
            d_az, d_el, d_work);
    oskar_convert_enu_direction_cosines_to_az_el_cudak_d
    OSKAR_CUDAK_CONF(n_blk, n_thd) (n, d_az, d_el, d_work, d_az, d_el);
    cudaDeviceSynchronize();
    return (int)cudaPeekAtLastError();
}

#ifdef __cplusplus
}
#endif
