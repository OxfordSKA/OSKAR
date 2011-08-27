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

#include "sky/oskar_cuda_transform_to_local_stokes.h"
#include "sky/cudak/oskar_cudak_transform_to_local_stokes.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single precision.
int oskar_cuda_transform_to_local_stokes_f(int ns, const float* d_ra,
        const float* d_dec, float lst, float lat, float* d_Q, float* d_U)
{
    // Precompute latitude trigonometry.
    float cos_lat = cosf(lat);
    float sin_lat = sinf(lat);

    // Set up thread and block dimensions.
    const int n_thd = 256;
    const int n_blk = (ns + n_thd - 1) / n_thd;

    // Compute the local Stokes parameters.
    oskar_cudak_transform_to_local_stokes_f <<< n_blk, n_thd >>> (ns,
            d_ra, d_dec, cos_lat, sin_lat, lst, d_Q, d_U);
    cudaDeviceSynchronize();

    return cudaPeekAtLastError();
}

// Double precision.
int oskar_cuda_transform_to_local_stokes_d(int ns, const double* d_ra,
        const double* d_dec, double lst, double lat, double* d_Q, double* d_U)
{
    // Precompute latitude trigonometry.
    double cos_lat = cos(lat);
    double sin_lat = sin(lat);

    // Set up thread and block dimensions.
    const int n_thd = 256;
    const int n_blk = (ns + n_thd - 1) / n_thd;

    // Compute the local Stokes parameters.
    oskar_cudak_transform_to_local_stokes_d <<< n_blk, n_thd >>> (ns,
            d_ra, d_dec, cos_lat, sin_lat, lst, d_Q, d_U);
    cudaDeviceSynchronize();

    return cudaPeekAtLastError();
}

#ifdef __cplusplus
}
#endif
