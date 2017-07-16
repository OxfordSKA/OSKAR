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

#include "math/oskar_dftw_o2c_3d_cuda.h"

/* Kernels. ================================================================ */

/* Shared memory pointers used by the kernels. */
extern __shared__ float2 smem_f[];
extern __shared__ double2 smem_d[];

/* Single precision. */
/* Value for max_in_chunk should be 800 in single precision. */
__global__
void oskar_dftw_o2c_3d_cudak_f(const int n_in,
        const float wavenumber,
        const float* __restrict__ x_in,
        const float* __restrict__ y_in,
        const float* __restrict__ z_in,
        const float2* __restrict__ weights_in,
        const int n_out,
        const float* __restrict__ x_out,
        const float* __restrict__ y_out,
        const float* __restrict__ z_out,
        const int max_in_chunk,
        float2* __restrict__ output)
{
    // Get the output position (pixel) ID that this thread is working on.
    const int i_out = blockDim.x * blockIdx.x + threadIdx.x;
    float2 out = make_float2(0.0f, 0.0f); // Clear output value.

    // Get the output position.
    // (NB. Cannot exit on index condition, as all threads are needed later.)
    float xp_out = 0.0f, yp_out = 0.0f, zp_out = 0.0f;
    if (i_out < n_out)
    {
        xp_out = wavenumber * x_out[i_out];
        yp_out = wavenumber * y_out[i_out];
        zp_out = wavenumber * z_out[i_out];
    }

    // Initialise shared memory caches.
    // Input positions are cached as float2 for speed increase.
    float2* cw = smem_f; // Cached input weights.
    float2* cp = cw + max_in_chunk; // Cached input x,y positions.
    float* cz = (float*)(cp + max_in_chunk); // Cached input z positions.

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
            cz[t] = z_in[g];
        }

        // Must synchronise before computing partial output for these inputs.
        __syncthreads();

        // Loop over input chunk.
        for (int i = 0; i < chunk_size; ++i)
        {
            // Calculate the phase for the output position.
            float2 signal, w = cw[i];
            float phase = xp_out * cp[i].x + yp_out * cp[i].y + zp_out * cz[i];
            sincosf(phase, &signal.y, &signal.x);

            // Perform complex multiply-accumulate.
            out.x += signal.x * w.x;
            out.x -= signal.y * w.y;
            out.y += signal.y * w.x;
            out.y += signal.x * w.y;
        }

        // Must synchronise again before loading in a new input chunk.
        __syncthreads();
    }

    // Copy result into global memory.
    if (i_out < n_out)
        output[i_out] = out;
}

/* Double precision. */
/* Value for max_in_chunk should be 384 in double precision. */
__global__
void oskar_dftw_o2c_3d_cudak_d(const int n_in,
        const double wavenumber,
        const double* __restrict__ x_in,
        const double* __restrict__ y_in,
        const double* __restrict__ z_in,
        const double2* __restrict__ weights_in,
        const int n_out,
        const double* __restrict__ x_out,
        const double* __restrict__ y_out,
        const double* __restrict__ z_out,
        const int max_in_chunk,
        double2* __restrict__ output)
{
    // Get the output position (pixel) ID that this thread is working on.
    const int i_out = blockDim.x * blockIdx.x + threadIdx.x;
    double2 out = make_double2(0.0, 0.0); // Clear output value.

    // Get the output position.
    // (NB. Cannot exit on index condition, as all threads are needed later.)
    double xp_out = 0.0, yp_out = 0.0, zp_out = 0.0;
    if (i_out < n_out)
    {
        xp_out = wavenumber * x_out[i_out];
        yp_out = wavenumber * y_out[i_out];
        zp_out = wavenumber * z_out[i_out];
    }

    // Initialise shared memory caches.
    double2* cw = smem_d; // Cached input weights.
    double2* cp = cw + max_in_chunk; // Cached input x,y positions.
    double* cz = (double*)(cp + max_in_chunk); // Cached input z positions.

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
            cz[t] = z_in[g];
        }

        // Must synchronise before computing partial output for these inputs.
        __syncthreads();

        // Loop over input chunk.
        for (int i = 0; i < chunk_size; ++i)
        {
            // Calculate the phase for the output position.
            double2 signal, w = cw[i];
            double phase = xp_out * cp[i].x + yp_out * cp[i].y + zp_out * cz[i];
            sincos(phase, &signal.y, &signal.x);

            // Perform complex multiply-accumulate.
            out.x += signal.x * w.x;
            out.x -= signal.y * w.y;
            out.y += signal.y * w.x;
            out.y += signal.x * w.y;
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
void oskar_dftw_o2c_3d_cuda_f(int n_in, float wavenumber, const float* d_x_in,
        const float* d_y_in, const float* d_z_in, const float2* d_weights_in,
        int n_out, const float* d_x_out, const float* d_y_out,
        const float* d_z_out, float2* d_output)
{
    int num_blocks, num_threads = 256;
    int shared_mem, max_in_chunk = 800; /* Should be multiple of 16. */
    num_blocks = (n_out + num_threads - 1) / num_threads;
    shared_mem = 5 * max_in_chunk * sizeof(float);
    oskar_dftw_o2c_3d_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem) (n_in, wavenumber,
            d_x_in, d_y_in, d_z_in, d_weights_in, n_out, d_x_out, d_y_out,
            d_z_out, max_in_chunk, d_output);
}

/* Double precision. */
void oskar_dftw_o2c_3d_cuda_d(int n_in, double wavenumber, const double* d_x_in,
        const double* d_y_in, const double* d_z_in, const double2* d_weights_in,
        int n_out, const double* d_x_out, const double* d_y_out,
        const double* d_z_out, double2* d_output)
{
    int num_blocks, num_threads = 256;
    int shared_mem, max_in_chunk = 384; /* Should be multiple of 16. */
    num_blocks = (n_out + num_threads - 1) / num_threads;
    shared_mem = 5 * max_in_chunk * sizeof(double);
    oskar_dftw_o2c_3d_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem) (n_in, wavenumber,
            d_x_in, d_y_in, d_z_in, d_weights_in, n_out, d_x_out, d_y_out,
            d_z_out, max_in_chunk, d_output);
}

#ifdef __cplusplus
}
#endif
