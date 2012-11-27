/*
 * Copyright (c) 2012, The University of Oxford
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

#include "sky/oskar_rebin_sky_cuda.h"
#include <stdio.h>

__global__
void oskar_rebin_sky_cudak_f(const int num_in, const int num_out,
        const float* lon_in, const float* lat_in, const float* flux_in,
        const float* lon_out, const float* lat_out, float* flux_out);

#ifdef __cplusplus
extern "C" {
#endif


/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_rebin_sky_cuda_f(int num_in, int num_out, const float* lon_in,
        const float* lat_in, const float* flux_in, const float* lon_out,
        const float* lat_out, float* flux_out)
{
    int num_chunks, max_chunk_size = 8192;

    /* Set up number of chunks. */
    num_chunks = (num_in + max_chunk_size - 1) / max_chunk_size;
    for (int i = 0; i < num_chunks; ++i)
    {
        int chunk_size, chunk_start, num_blocks, num_threads = 256;

        /* Get chunk start ID, and chunk size. */
        chunk_start = i * max_chunk_size;
        chunk_size = num_in - chunk_start;
        if (chunk_size > max_chunk_size)
            chunk_size = max_chunk_size;

        /* Set up thread blocks and call the kernel. */
        num_blocks = (chunk_size + num_threads - 1) / num_threads;
        oskar_rebin_sky_cudak_f OSKAR_CUDAK_CONF(num_blocks, num_threads)
        (chunk_size, num_out, &lon_in[chunk_start], &lat_in[chunk_start],
                &flux_in[chunk_start], lon_out, lat_out, flux_out);
    }
}

#ifdef __cplusplus
}
#endif


/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_rebin_sky_cudak_f(const int num_in, const int num_out,
        const float* lon_in, const float* lat_in, const float* flux_in,
        const float* lon_out, const float* lat_out, float* flux_out)
{
    /* Get the input source ID that this thread is working on. */
    const int s = blockDim.x * blockIdx.x + threadIdx.x;
    if (s >= num_in) return;

    /* Save some input parameters. */
    const float lat_in_s = lat_in[s];
    const float lon_in_s = lon_in[s];
    const float cos_lat_in_s = cosf(lat_in_s);

    /* Maintain minimum separation. */
    float min_sep = 10.0f; /* Radians. */
    int min_sep_index = 0;

    /* Loop over output source positions. */
    for (int i = 0; i < num_out; ++i)
    {
        /* Compute angular separation. */
        float sin_delta_lat = sinf(0.5f * (lat_in_s - lat_out[i]));
        float sin_delta_lon = sinf(0.5f * (lon_in_s - lon_out[i]));
        float cos_lat_out = cosf(lat_out[i]);
        float delta = 2.0f * asinf(sqrtf(sin_delta_lat*sin_delta_lat +
                cos_lat_out * cos_lat_in_s * sin_delta_lon*sin_delta_lon));

        /* Check if this is a new minimum separation. */
        if (delta < min_sep)
        {
            min_sep = delta;
            min_sep_index = i;
        }
    }

    /* Atomic add input flux to output point. */
    atomicAdd(&flux_out[min_sep_index], flux_in[s]);
}

