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

#include "math/oskar_cuda_dft_c2r_2d.h"
#include "math/cudak/oskar_cudak_dft_c2r_2d.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single precision.
int oskar_cuda_dft_c2r_2d_f(int n_in, const float* x_in, const float* y_in,
        const float* data_in, int n_out, const float* x_out,
        const float* y_out, float* output)
{
    // Initialise.
    const int thd = 384; // Threads per block.
    const int max_in_chunk = 896; // Should be multiple of 16.
    const int max_out_chunk = 65536; // Manageable output chunk size.
    const size_t shared = 2 * max_in_chunk * sizeof(float2);
    const float2* data = (const float2*)data_in;

    // Loop over output chunks.
    for (int start = 0; start < n_out; start += max_out_chunk)
    {
        int chunk_size = n_out - start;
        if (chunk_size > max_out_chunk) chunk_size = max_out_chunk;

        // Invoke kernel to compute the (partial) DFT on the device.
        const int blk = (chunk_size + thd - 1) / thd;
        oskar_cudak_dft_c2r_2d_f <<< blk, thd, shared >>>
                (n_in, x_in, y_in, data, chunk_size, x_out + start,
                        y_out + start, max_in_chunk, output + start);
        cudaThreadSynchronize();
        cudaError_t errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) return errCuda;
    }

    return 0;
}

// Double precision.
int oskar_cuda_dft_c2r_2d_d(int n_in, const double* x_in, const double* y_in,
        const double* data_in, int n_out, const double* x_out,
        const double* y_out, double* output)
{
    // Initialise.
    const int thd = 384; // Threads per block.
    const int max_in_chunk = 448; // Should be multiple of 16.
    const int max_out_chunk = 32768; // Manageable output chunk size.
    const size_t shared = 2 * max_in_chunk * sizeof(double2);
    const double2* data = (const double2*)data_in;

    // Loop over output chunks.
    for (int start = 0; start < n_out; start += max_out_chunk)
    {
        int chunk_size = n_out - start;
        if (chunk_size > max_out_chunk) chunk_size = max_out_chunk;

        // Invoke kernel to compute the (partial) DFT on the device.
        const int blk = (chunk_size + thd - 1) / thd;
        oskar_cudak_dft_c2r_2d_d <<< blk, thd, shared >>>
                (n_in, x_in, y_in, data, chunk_size, x_out + start,
                        y_out + start, max_in_chunk, output + start);
        cudaThreadSynchronize();
        cudaError_t errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) return errCuda;
    }

    return 0;
}

#ifdef __cplusplus
}
#endif
