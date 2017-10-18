/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include "math/oskar_dftw_m2m_2d_cuda.h"

/* Kernels. ================================================================ */

/* Shared memory pointers used by the kernels. */
extern __shared__ float2 smem_f[];
extern __shared__ double2 smem_d[];

/* Single precision. */
__global__
void oskar_dftw_m2m_2d_cudak_f(const int n_in,
        const float wavenumber,
        const float* __restrict__ x_in,
        const float* __restrict__ y_in,
        const float2* __restrict__ weights_in,
        const int n_out,
        const float* __restrict__ x_out,
        const float* __restrict__ y_out,
        const int max_in_chunk,
        const float4c* __restrict__ data,
        float4c* __restrict__ output)
{
    // Get the output position (pixel) ID that this thread is working on.
    const int i_out = blockDim.x * blockIdx.x + threadIdx.x;

    // Clear output value.
    float4c out;
    out.a = make_float2(0.0f, 0.0f);
    out.b = make_float2(0.0f, 0.0f);
    out.c = make_float2(0.0f, 0.0f);
    out.d = make_float2(0.0f, 0.0f);

    // Get the output position.
    // (NB. Cannot exit on index condition, as all threads are needed later.)
    float xp_out = 0.0f, yp_out = 0.0f;
    if (i_out < n_out)
    {
        xp_out = wavenumber * x_out[i_out];
        yp_out = wavenumber * y_out[i_out];
    }

    // Initialise shared memory caches.
    // Input positions are cached as float2 for speed increase.
    float2* cw = smem_f; // Cached input weights.
    float2* cp = cw + max_in_chunk; // Cached input positions.

    // Cache a chunk of input positions and weights into shared memory.
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
            cw[t] = weights_in[g];
            cp[t].x = x_in[g];
            cp[t].y = y_in[g];
        }

        // Must synchronise before computing partial output for these inputs.
        __syncthreads();

        // Loop over input chunk.
        if (i_out < n_out)
        {
            for (int i = 0; i < chunk_size; ++i)
            {
                // Calculate the DFT phase for the output position.
                float2 weight;
                {
                    float t = xp_out * cp[i].x + yp_out * cp[i].y; // Phase.
                    sincosf(t, &weight.y, &weight.x);

                    // Multiply the DFT phase weight by the supplied weight.
                    float2 w = cw[i];
                    t = weight.x; // Re-use register and copy the real part.
                    weight.x *= w.x;
                    weight.x -= w.y * weight.y;
                    weight.y *= w.x;
                    weight.y += w.y * t;
                }

                // Complex multiply-accumulate input signal and weight.
                float4c in = data[(start + i) * n_out + i_out];
                out.a.x += in.a.x * weight.x;
                out.a.x -= in.a.y * weight.y;
                out.a.y += in.a.y * weight.x;
                out.a.y += in.a.x * weight.y;
                out.b.x += in.b.x * weight.x;
                out.b.x -= in.b.y * weight.y;
                out.b.y += in.b.y * weight.x;
                out.b.y += in.b.x * weight.y;
                out.c.x += in.c.x * weight.x;
                out.c.x -= in.c.y * weight.y;
                out.c.y += in.c.y * weight.x;
                out.c.y += in.c.x * weight.y;
                out.d.x += in.d.x * weight.x;
                out.d.x -= in.d.y * weight.y;
                out.d.y += in.d.y * weight.x;
                out.d.y += in.d.x * weight.y;
            }
        }

        // Must synchronise again before loading in a new input chunk.
        __syncthreads();
    }

    // Copy result into global memory.
    if (i_out < n_out)
        output[i_out] = out;
}

/* Double precision. */
__global__
void oskar_dftw_m2m_2d_cudak_d(const int n_in,
        const double wavenumber,
        const double* __restrict__ x_in,
        const double* __restrict__ y_in,
        const double2* __restrict__ weights_in,
        const int n_out,
        const double* __restrict__ x_out,
        const double* __restrict__ y_out,
        const int max_in_chunk,
        const double4c* __restrict__ data,
        double4c* __restrict__ output)
{
    // Get the output position (pixel) ID that this thread is working on.
    const int i_out = blockDim.x * blockIdx.x + threadIdx.x;

    // Clear output value.
    double4c out;
    out.a = make_double2(0.0, 0.0);
    out.b = make_double2(0.0, 0.0);
    out.c = make_double2(0.0, 0.0);
    out.d = make_double2(0.0, 0.0);

    // Get the output position.
    // (NB. Cannot exit on index condition, as all threads are needed later.)
    double xp_out = 0.0, yp_out = 0.0;
    if (i_out < n_out)
    {
        xp_out = wavenumber * x_out[i_out];
        yp_out = wavenumber * y_out[i_out];
    }

    // Initialise shared memory caches.
    // Input positions are cached as double2 for speed increase.
    double2* cw = smem_d; // Cached input weights.
    double2* cp = cw + max_in_chunk; // Cached input positions.

    // Cache a chunk of input positions and weights into shared memory.
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
            cw[t] = weights_in[g];
            cp[t].x = x_in[g];
            cp[t].y = y_in[g];
        }

        // Must synchronise before computing partial output for these inputs.
        __syncthreads();

        // Loop over input chunk.
        if (i_out < n_out)
        {
            for (int i = 0; i < chunk_size; ++i)
            {
                // Calculate the DFT phase for the output position.
                double2 weight;
                {
                    double t = xp_out * cp[i].x + yp_out * cp[i].y; // Phase.
                    sincos(t, &weight.y, &weight.x);

                    // Multiply the DFT phase weight by the supplied weight.
                    double2 w = cw[i];
                    t = weight.x; // Re-use register and copy the real part.
                    weight.x *= w.x;
                    weight.x -= w.y * weight.y;
                    weight.y *= w.x;
                    weight.y += w.y * t;
                }

                // Complex multiply-accumulate input signal and weight.
                double4c in = data[(start + i) * n_out + i_out];
                out.a.x += in.a.x * weight.x;
                out.a.x -= in.a.y * weight.y;
                out.a.y += in.a.y * weight.x;
                out.a.y += in.a.x * weight.y;
                out.b.x += in.b.x * weight.x;
                out.b.x -= in.b.y * weight.y;
                out.b.y += in.b.y * weight.x;
                out.b.y += in.b.x * weight.y;
                out.c.x += in.c.x * weight.x;
                out.c.x -= in.c.y * weight.y;
                out.c.y += in.c.y * weight.x;
                out.c.y += in.c.x * weight.y;
                out.d.x += in.d.x * weight.x;
                out.d.x -= in.d.y * weight.y;
                out.d.y += in.d.y * weight.x;
                out.d.y += in.d.x * weight.y;
            }
        }

        // Must synchronise again before loading in a new input chunk.
        __syncthreads();
    }

    // Copy result into global memory.
    if (i_out < n_out)
        output[i_out] = out;
}


#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_dftw_m2m_2d_cuda_f(int n_in, float wavenumber, const float* d_x_in,
        const float* d_y_in, const float2* d_weights_in, int n_out,
        const float* d_x_out, const float* d_y_out, const float4c* d_data,
        float4c* d_output)
{
    int num_blocks, num_threads = 256;
    int shared_mem, max_in_chunk = 896; /* Should be multiple of 16. */
    num_blocks = (n_out + num_threads - 1) / num_threads;
    shared_mem = 4 * max_in_chunk * sizeof(float);
    oskar_dftw_m2m_2d_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem) (n_in, wavenumber,
            d_x_in, d_y_in, d_weights_in, n_out, d_x_out, d_y_out,
            max_in_chunk, d_data, d_output);
}

/* Double precision. */
void oskar_dftw_m2m_2d_cuda_d(int n_in, double wavenumber, const double* d_x_in,
        const double* d_y_in, const double2* d_weights_in, int n_out,
        const double* d_x_out, const double* d_y_out, const double4c* d_data,
        double4c* d_output)
{
    int num_blocks, num_threads = 256;
    int shared_mem, max_in_chunk = 448; /* Should be multiple of 16. */
    num_blocks = (n_out + num_threads - 1) / num_threads;
    shared_mem = 4 * max_in_chunk * sizeof(double);
    oskar_dftw_m2m_2d_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem) (n_in, wavenumber,
            d_x_in, d_y_in, d_weights_in, n_out, d_x_out, d_y_out,
            max_in_chunk, d_data, d_output);
}

#ifdef __cplusplus
}
#endif
