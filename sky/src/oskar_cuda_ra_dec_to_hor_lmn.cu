/*
 * Copyright (c) 2011, The University of Oxford
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

#include "sky/oskar_cuda_ra_dec_to_hor_lmn.h"
#include "sky/cudak/oskar_cudak_ha_dec_to_hor_lmn.h"
#include "math/cudak/oskar_math_cudak_vec_sub_sr.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single precision.

int oskar_cuda_ra_dec_to_hor_lmn_f(int n, const float* ra,
        const float* dec, float lst, float lat, float* hor_l, float* hor_m,
        float* hor_n)
{
    // Determine source Hour Angles (HA = LST - RA).
    float* ha = hor_n; // Temporary.
    const int n_thd = 256;
    const int n_blk_in = (n + n_thd - 1) / n_thd;
    oskar_math_cudakf_vec_sub_sr <<< n_blk_in, n_thd >>> (n, lst, ra, ha);

    // Determine horizontal l,m,n positions (destroys contents of ha).
    float cosLat = cosf(lat);
    float sinLat = sinf(lat);
    oskar_cudak_ha_dec_to_hor_lmn_f <<< n_blk_in, n_thd >>> (n, ha, dec,
            cosLat, sinLat, hor_l, hor_m, hor_n);
    cudaDeviceSynchronize();
    cudaError_t errCuda = cudaPeekAtLastError();
    if (errCuda != cudaSuccess) return errCuda;

    return 0;
}

// Double precision.

int oskar_cuda_ra_dec_to_hor_lmn_d(int n, const double* ra,
        const double* dec, double lst, double lat, double* hor_l, double* hor_m,
        double* hor_n)
{
    // Determine source Hour Angles (HA = LST - RA).
    double* ha = hor_n; // Temporary.
    const int n_thd = 256;
    const int n_blk_in = (n + n_thd - 1) / n_thd;
    oskar_math_cudakd_vec_sub_sr <<< n_blk_in, n_thd >>> (n, lst, ra, ha);

    // Determine horizontal l,m,n positions (destroys contents of ha).
    double cosLat = cos(lat);
    double sinLat = sin(lat);
    oskar_cudak_ha_dec_to_hor_lmn_d <<< n_blk_in, n_thd >>> (n, ha, dec,
            cosLat, sinLat, hor_l, hor_m, hor_n);
    cudaDeviceSynchronize();
    cudaError_t errCuda = cudaPeekAtLastError();
    if (errCuda != cudaSuccess) return errCuda;

    return 0;
}

#ifdef __cplusplus
}
#endif
