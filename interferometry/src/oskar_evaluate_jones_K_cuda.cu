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

#include <oskar_evaluate_jones_K_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_evaluate_jones_K_cuda_f(float2* d_jones, int num_stations,
        float wavenumber, const float* d_u, const float* d_v, const float* d_w,
        int num_sources, const float* d_l, const float* d_m, const float* d_n)
{
    /* Define block and grid sizes. */
    const dim3 num_threads(64, 4); /* Sources, stations. */
    const dim3 num_blocks((num_sources + num_threads.x - 1) / num_threads.x,
            (num_stations + num_threads.y - 1) / num_threads.y);
    const size_t s_mem = 3 * (num_threads.x + num_threads.y) * sizeof(float);

    /* Compute DFT phase weights for K. */
    oskar_evaluate_jones_K_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads, s_mem)
    (num_stations, wavenumber, d_u, d_v, d_w, num_sources, d_l, d_m, d_n,
            d_jones);
}

/* Double precision. */
void oskar_evaluate_jones_K_cuda_d(double2* d_jones, int num_stations,
        double wavenumber, const double* d_u, const double* d_v,
        const double* d_w, int num_sources, const double* d_l,
        const double* d_m, const double* d_n)
{
    /* Define block and grid sizes. */
    const dim3 num_threads(64, 4); /* Sources, stations. */
    const dim3 num_blocks((num_sources + num_threads.x - 1) / num_threads.x,
            (num_stations + num_threads.y - 1) / num_threads.y);
    const size_t s_mem = 3 * (num_threads.x + num_threads.y) * sizeof(double);

    /* Compute DFT phase weights for K. */
    oskar_evaluate_jones_K_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads, s_mem)
    (num_stations, wavenumber, d_u, d_v, d_w, num_sources, d_l, d_m, d_n,
            d_jones);
}

#ifdef __cplusplus
}
#endif


/* Kernels. ================================================================ */

/* Shared memory pointers used by the kernels. */
extern __shared__ float smem_f[];
extern __shared__ double smem_d[];

/* Single precision. */
__global__
void oskar_evaluate_jones_K_cudak_f(const int n_in, const float wavenumber,
        const float* x_in, const float* y_in, const float* z_in,
        const int n_out, const float* x_out, const float* y_out,
        const float* z_out, float2* weights)
{
    const int s = blockDim.x * blockIdx.x + threadIdx.x; /* Output index. */
    const int a = blockDim.y * blockIdx.y + threadIdx.y; /* Input index. */

    /* Cache input and output data from global memory. */
    float* cxo = smem_f;
    float* cyo = &cxo[blockDim.x];
    float* czo = &cyo[blockDim.x];
    float* cxi = &czo[blockDim.x];
    float* cyi = &cxi[blockDim.y];
    float* czi = &cyi[blockDim.y];
    if (s < n_out && threadIdx.y == 0)
    {
        cxo[threadIdx.x] = x_out[s];
        cyo[threadIdx.x] = y_out[s];
        czo[threadIdx.x] = z_out[s];
    }
    if (a < n_in && threadIdx.x == 0)
    {
        cxi[threadIdx.y] = wavenumber * x_in[a];
        cyi[threadIdx.y] = wavenumber * y_in[a];
        czi[threadIdx.y] = wavenumber * z_in[a];
    }
    __syncthreads();

    /* Compute the geometric phase of the output direction. */
    float phase;
    phase =  cxi[threadIdx.y] * cxo[threadIdx.x];
    phase += cyi[threadIdx.y] * cyo[threadIdx.x];
    phase += czi[threadIdx.y] * czo[threadIdx.x];
    float2 weight;
    sincosf(phase, &weight.y, &weight.x);

    /* Write result to global memory. */
    if (s < n_out && a < n_in)
    {
        const int w = s + n_out * a;
        weights[w] = weight;
    }
}

/* Double precision. */
__global__
void oskar_evaluate_jones_K_cudak_d(const int n_in, const double wavenumber,
        const double* x_in, const double* y_in, const double* z_in,
        const int n_out, const double* x_out, const double* y_out,
        const double* z_out, double2* weights)
{
    const int s = blockDim.x * blockIdx.x + threadIdx.x; /* Output index. */
    const int a = blockDim.y * blockIdx.y + threadIdx.y; /* Input index. */

    /* Cache input and output data from global memory. */
    double* cxo = smem_d;
    double* cyo = &cxo[blockDim.x];
    double* czo = &cyo[blockDim.x];
    double* cxi = &czo[blockDim.x];
    double* cyi = &cxi[blockDim.y];
    double* czi = &cyi[blockDim.y];
    if (s < n_out && threadIdx.y == 0)
    {
        cxo[threadIdx.x] = x_out[s];
        cyo[threadIdx.x] = y_out[s];
        czo[threadIdx.x] = z_out[s];
    }
    if (a < n_in && threadIdx.x == 0)
    {
        cxi[threadIdx.y] = wavenumber * x_in[a];
        cyi[threadIdx.y] = wavenumber * y_in[a];
        czi[threadIdx.y] = wavenumber * z_in[a];
    }
    __syncthreads();

    /* Compute the geometric phase of the output direction. */
    double phase;
    phase =  cxi[threadIdx.y] * cxo[threadIdx.x];
    phase += cyi[threadIdx.y] * cyo[threadIdx.x];
    phase += czi[threadIdx.y] * czo[threadIdx.x];
    double2 weight;
    sincos(phase, &weight.y, &weight.x);

    /* Write result to global memory. */
    if (s < n_out && a < n_in)
    {
        const int w = s + n_out * a;
        weights[w] = weight;
    }
}
