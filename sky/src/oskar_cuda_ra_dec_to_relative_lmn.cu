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

#include "sky/oskar_cuda_ra_dec_to_relative_lmn.h"
#include "math/cudak/oskar_math_cudak_sph_to_lm.h"
#include "sky/cudak/oskar_cudak_lm_to_n.h"

// Single precision.

int oskar_cuda_ra_dec_to_relative_lmn_f(int n, const float* ra,
        const float* dec, float ra0, float dec0, float* p_l, float* p_m,
        float* p_n)
{
    // Compute l,m-direction-cosines of RA, Dec relative to reference point.
    const int n_thd = 256;
    const int n_blk = (n + n_thd - 1) / n_thd;
    const float cosDec0 = cosf(dec0);
    const float sinDec0 = sinf(dec0);
    oskar_math_cudakf_sph_to_lm <<< n_blk, n_thd >>>
            (n, ra, dec, ra0, cosDec0, sinDec0, p_l, p_m);
    cudaDeviceSynchronize();
    cudaError_t errCuda = cudaPeekAtLastError();
    if (errCuda != cudaSuccess) return errCuda;

    // Compute n-direction-cosines of points from l and m.
    oskar_cudak_lm_to_n_f <<< n_blk, n_thd >>> (n, p_l, p_m, p_n);
    cudaDeviceSynchronize();
    errCuda = cudaPeekAtLastError();
    if (errCuda != cudaSuccess) return errCuda;

    return 0;
}

// Double precision.

int oskar_cuda_ra_dec_to_relative_lmn_d(int n, const double* ra,
        const double* dec, double ra0, double dec0, double* p_l, double* p_m,
        double* p_n)
{
    // Compute l,m-direction-cosines of RA, Dec relative to reference point.
    const int n_thd = 256;
    const int n_blk = (n + n_thd - 1) / n_thd;
    const double cosDec0 = cos(dec0);
    const double sinDec0 = sin(dec0);
    oskar_math_cudakd_sph_to_lm <<< n_blk, n_thd >>>
            (n, ra, dec, ra0, cosDec0, sinDec0, p_l, p_m);
    cudaDeviceSynchronize();
    cudaError_t errCuda = cudaPeekAtLastError();
    if (errCuda != cudaSuccess) return errCuda;

    // Compute n-direction-cosines of points from l and m.
    oskar_cudak_lm_to_n_d <<< n_blk, n_thd >>> (n, p_l, p_m, p_n);
    cudaDeviceSynchronize();
    errCuda = cudaPeekAtLastError();
    if (errCuda != cudaSuccess) return errCuda;

    return 0;
}
