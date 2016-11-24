/*
 * Copyright (c) 2011-2016, The University of Oxford
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

#include "math/oskar_dft_c2r_2d_cuda.h"


/* Kernels. ================================================================ */

/* Shared memory pointers used by the kernels. */
extern __shared__ float4 smem_f[];
extern __shared__ double4 smem_d[];

/* Single precision. */
__global__
void oskar_dft_c2r_2d_cudak_f(int n_in,
        const float wavenumber,
        const float* __restrict__ x_in,
        const float* __restrict__ y_in,
        const float2* __restrict__ data_in,
        const float* __restrict__ weight_in,
        const int n_out,
        const float* __restrict__ x_out,
        const float* __restrict__ y_out,
        const int max_in_chunk,
        float* __restrict__ output)
{
    // Get the output position (pixel) ID that this thread is working on.
    const int i_out = blockDim.x * blockIdx.x + threadIdx.x;
    float out = 0.0f; // Clear output value.

    // Get the output position.
    // (NB. Cannot exit on index condition, as all threads are needed later.)
    float xp_out = 0.0f, yp_out = 0.0f;
    if (i_out < n_out)
    {
        xp_out = wavenumber * x_out[i_out];
        yp_out = wavenumber * y_out[i_out];
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
            smem_f[t].x = x_in[g];
            smem_f[t].y = y_in[g];
            smem_f[t].z = data_in[g].x * weight_in[g];
            smem_f[t].w = data_in[g].y * weight_in[g];
        }

        // Must synchronise before computing partial output for these inputs.
        __syncthreads();

        // Loop over input block.
        for (int i = 0; i < chunk_size; ++i)
        {
            // Calculate the complex DFT weight.
            float2 weight;
            float a = smem_f[i].x * xp_out + smem_f[i].y * yp_out;
            sincosf(-a, &weight.y, &weight.x);

            // Perform complex multiply-accumulate.
            // Output is real, so only evaluate the real part.
            out += smem_f[i].z * weight.x; // RE*RE
            out -= smem_f[i].w * weight.y; // IM*IM
        }

        // Must synchronise again before loading in a new input block.
        __syncthreads();
    }

    // Copy result into global memory.
    if (i_out < n_out)
        output[i_out] = out;
}

/* Double precision. */
__global__
void oskar_dft_c2r_2d_cudak_d(int n_in,
        const double wavenumber,
        const double* __restrict__ x_in,
        const double* __restrict__ y_in,
        const double2* __restrict__ data_in,
        const double* __restrict__ weight_in,
        const int n_out,
        const double* __restrict__ x_out,
        const double* __restrict__ y_out,
        const int max_in_chunk,
        double* __restrict__ output)
{
    // Get the output position (pixel) ID that this thread is working on.
    const int i_out = blockDim.x * blockIdx.x + threadIdx.x;
    double out = 0.0; // Clear output value.

    // Get the output position.
    // (NB. Cannot exit on index condition, as all threads are needed later.)
    double xp_out = 0.0, yp_out = 0.0;
    if (i_out < n_out)
    {
        xp_out = wavenumber * x_out[i_out];
        yp_out = wavenumber * y_out[i_out];
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
            smem_d[t].x = x_in[g];
            smem_d[t].y = y_in[g];
            smem_d[t].z = data_in[g].x * weight_in[g];
            smem_d[t].w = data_in[g].y * weight_in[g];
        }

        // Must synchronise before computing partial output for these inputs.
        __syncthreads();

        // Loop over input block.
        for (int i = 0; i < chunk_size; ++i)
        {
            // Calculate the complex DFT weight.
            double2 weight;
            double a = smem_d[i].x * xp_out + smem_d[i].y * yp_out;
            sincos(-a, &weight.y, &weight.x);

            // Perform complex multiply-accumulate.
            // Output is real, so only evaluate the real part.
            out += smem_d[i].z * weight.x; // RE*RE
            out -= smem_d[i].w * weight.y; // IM*IM
        }

        // Must synchronise again before loading in a new input block.
        __syncthreads();
    }

    // Copy result into global memory.
    if (i_out < n_out)
        output[i_out] = out;
}

#ifdef __cplusplus
extern "C" {
#endif

/* Utility functions. */
static int oskar_int_round_to_nearest_multiple(int num_to_round, int multiple)
{
   return (num_to_round + multiple - 1) / multiple * multiple;
}

static int oskar_int_range_clamp(int value, int minimum, int maximum)
{
   if (value < minimum)
       return minimum;
   if (value > maximum)
       return maximum;
   return value;
}


/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_dft_c2r_2d_cuda_f(int num_in, float wavenumber, const float* x_in,
        const float* y_in, const float2* data_in, const float* weight_in,
        int num_out, const float* x_out, const float* y_out, float* output)
{
    const int threads = 384;     /* Should be multiple of 32. */
    const int max_in_size = 896; /* Should be multiple of 16. */
    int out_size, max_out_size, blocks, shared_mem_size, start;

    /* Initialise. */
    shared_mem_size = 2 * max_in_size * sizeof(float2);

    /* Compute the maximum manageable output chunk size. */
    max_out_size = 65536 * 8192; /* Product of max output and input sizes. */
    max_out_size /= num_in;
    max_out_size = oskar_int_round_to_nearest_multiple(max_out_size, threads);
    max_out_size = oskar_int_range_clamp(max_out_size,
            2 * threads, 160 * threads);

    /* Loop over output chunks. */
    for (start = 0; start < num_out; start += max_out_size)
    {
        out_size = num_out - start;
        if (out_size > max_out_size) out_size = max_out_size;

        /* Invoke kernel to compute the (partial) DFT on the device. */
        blocks = (out_size + threads - 1) / threads;
        oskar_dft_c2r_2d_cudak_f
        OSKAR_CUDAK_CONF(blocks, threads, shared_mem_size) (num_in, wavenumber,
                x_in, y_in, data_in, weight_in, out_size,
                x_out + start, y_out + start, max_in_size, output + start);
    }
}

/* Double precision. */
void oskar_dft_c2r_2d_cuda_d(int num_in, double wavenumber, const double* x_in,
        const double* y_in, const double2* data_in, const double* weight_in,
        int num_out, const double* x_out, const double* y_out, double* output)
{
    const int threads = 384;     /* Should be multiple of 32. */
    const int max_in_size = 448; /* Should be multiple of 16. */
    int out_size, max_out_size, blocks, shared_mem_size, start;

    /* Initialise. */
    shared_mem_size = 2 * max_in_size * sizeof(double2);

    /* Compute the maximum manageable output chunk size. */
    max_out_size = 32768 * 8192; /* Product of max output and input sizes. */
    max_out_size /= num_in;
    max_out_size = oskar_int_round_to_nearest_multiple(max_out_size, threads);
    max_out_size = oskar_int_range_clamp(max_out_size,
            2 * threads, 80 * threads);

    /* Loop over output chunks. */
    for (start = 0; start < num_out; start += max_out_size)
    {
        out_size = num_out - start;
        if (out_size > max_out_size) out_size = max_out_size;

        /* Invoke kernel to compute the (partial) DFT on the device. */
        blocks = (out_size + threads - 1) / threads;
        oskar_dft_c2r_2d_cudak_d
        OSKAR_CUDAK_CONF(blocks, threads, shared_mem_size) (num_in, wavenumber,
                x_in, y_in, data_in, weight_in, out_size,
                x_out + start, y_out + start, max_in_size, output + start);
    }
}

#ifdef __cplusplus
}
#endif
