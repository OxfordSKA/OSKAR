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

#include "math/cudak/oskar_cudak_dft_c2r_2d.h"

// Shared memory pointer used by the kernel.
extern __shared__ float4 c[];

// Single precision.

__global__
void oskar_cudak_dft_c2r_2d_f(int n_in, const float* x_in,
        const float* y_in, const float2* data_in, const int n_out,
        const float* x_out, const float* y_out, const int max_in_chunk,
        float* output)
{
    // Get the output position (pixel) ID that this thread is working on.
    const int i_out = blockDim.x * blockIdx.x + threadIdx.x;
    float out = 0.0f; // Clear output value.

    // Get the output position.
    // (NB. Cannot exit on index condition, as all threads are needed later.)
    float xp_out = 0.0f, yp_out = 0.0f;
    if (i_out < n_out)
    {
        xp_out = x_out[i_out];
        yp_out = y_out[i_out];
    }

    // Cache a chunk of input data and positions into shared memory.
    for (int start = 0; start < n_in; start += max_in_chunk)
    {
        int chunk_size = n_in - start;
        if (chunk_size > max_in_chunk)
            chunk_size = max_in_chunk;

        // There are blockDim.x threads available - need to copy
        // chunk_size pieces of data from global memory.
        for (int t = threadIdx.x; t < chunk_size; t += blockDim.x)
        {
            const int g = start + t; // Global input index.
            c[t].x = x_in[g];
            c[t].y = y_in[g];
            c[t].z = data_in[g].x;
            c[t].w = data_in[g].y;
        }

        // Must synchronise before computing partial output for these inputs.
        __syncthreads();

        // Loop over input block.
        for (int i = 0; i < chunk_size; ++i)
        {
            // Calculate the complex DFT weight.
            float2 weight;
            float a = c[i].x * xp_out + c[i].y * yp_out;
            sincosf(a, &weight.y, &weight.x);

            // Perform complex multiply-accumulate.
            // Output is real, so only evaluate the real part.
            out += c[i].z * weight.x; // RE*RE
            out -= c[i].w * weight.y; // IM*IM
        }

        // Must synchronise again before loading in a new input block.
        __syncthreads();
    }

    // Copy result into global memory.
    if (i_out < n_out)
        output[i_out] = out;
}

// Shared memory pointer used by the kernel.
extern __shared__ double4 cd[];

// Double precision.

__global__
void oskar_cudak_dft_c2r_2d_d(int n_in, const double* x_in,
        const double* y_in, const double2* data_in, const int n_out,
        const double* x_out, const double* y_out, const int max_in_chunk,
        double* output)
{
    // Get the output position (pixel) ID that this thread is working on.
    const int i_out = blockDim.x * blockIdx.x + threadIdx.x;
    double out = 0.0; // Clear output value.

    // Get the output position.
    // (NB. Cannot exit on index condition, as all threads are needed later.)
    double xp_out = 0.0, yp_out = 0.0;
    if (i_out < n_out)
    {
        xp_out = x_out[i_out];
        yp_out = y_out[i_out];
    }

    // Cache a chunk of input data and positions into shared memory.
    for (int start = 0; start < n_in; start += max_in_chunk)
    {
        int chunk_size = n_in - start;
        if (chunk_size > max_in_chunk)
            chunk_size = max_in_chunk;

        // There are blockDim.x threads available - need to copy
        // chunk_size pieces of data from global memory.
        for (int t = threadIdx.x; t < chunk_size; t += blockDim.x)
        {
            const int g = start + t; // Global input index.
            cd[t].x = x_in[g];
            cd[t].y = y_in[g];
            cd[t].z = data_in[g].x;
            cd[t].w = data_in[g].y;
        }

        // Must synchronise before computing partial output for these inputs.
        __syncthreads();

        // Loop over input block.
        for (int i = 0; i < chunk_size; ++i)
        {
            // Calculate the complex DFT weight.
            double2 weight;
            double a = cd[i].x * xp_out + cd[i].y * yp_out;
            sincos(a, &weight.y, &weight.x);

            // Perform complex multiply-accumulate.
            // Output is real, so only evaluate the real part.
            out += cd[i].z * weight.x; // RE*RE
            out -= cd[i].w * weight.y; // IM*IM
        }

        // Must synchronise again before loading in a new input block.
        __syncthreads();
    }

    // Copy result into global memory.
    if (i_out < n_out)
        output[i_out] = out;
}
