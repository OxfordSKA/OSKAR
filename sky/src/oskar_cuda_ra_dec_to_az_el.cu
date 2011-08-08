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
#include "sky/cudak/oskar_cudak_ra_dec_to_hor_lmn.h"
#include "sky/cudak/oskar_cudak_hor_lmn_to_az_el.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single precision.

int oskar_cuda_ra_dec_to_az_el_f(int n, const float* ra,
        const float* dec, float lst, float lat, float* az, float* el,
        float* work)
{
    // Determine horizontal coordinates.
    const int n_thd = 512;
    const int n_blk = (n + n_thd - 1) / n_thd;
    float cosLat = cosf(lat);
    float sinLat = sinf(lat);
    oskar_cudak_ra_dec_to_hor_lmn_f <<< n_blk, n_thd >>> (n, ra, dec,
            cosLat, sinLat, lst, az, el, work);
    oskar_cudak_hor_lmn_to_az_el_f <<< n_blk, n_thd >>> (n, az, el, work,
            az, el);
    cudaDeviceSynchronize();
    cudaError_t errCuda = cudaPeekAtLastError();
    if (errCuda != cudaSuccess) return errCuda;

    return 0;
}

// Double precision.

int oskar_cuda_ra_dec_to_az_el_d(int n, const double* ra,
        const double* dec, double lst, double lat, double* az, double* el,
        double* work)
{
    // Determine horizontal coordinates.
    const int n_thd = 512;
    const int n_blk = (n + n_thd - 1) / n_thd;
    float cosLat = cos(lat);
    float sinLat = sin(lat);
    oskar_cudak_ra_dec_to_hor_lmn_d <<< n_blk, n_thd >>> (n, ra, dec,
            cosLat, sinLat, lst, az, el, work);
    oskar_cudak_hor_lmn_to_az_el_d <<< n_blk, n_thd >>> (n, az, el, work,
            az, el);
    cudaDeviceSynchronize();
    cudaError_t errCuda = cudaPeekAtLastError();
    if (errCuda != cudaSuccess) return errCuda;

    return 0;
}

#ifdef __cplusplus
}
#endif
