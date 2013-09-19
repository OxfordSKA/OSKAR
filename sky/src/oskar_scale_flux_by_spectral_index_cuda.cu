/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include <oskar_scale_flux_by_spectral_index_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_scale_flux_by_spectral_index_cuda_f(int num_sources,
        float frequency, float* d_I, float* d_Q, float* d_U, float* d_V,
        float* d_ref_freq, const float* d_sp_index)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_sources + num_threads - 1) / num_threads;
    oskar_scale_flux_by_spectral_index_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_sources, frequency,
            d_I, d_Q, d_U, d_V, d_ref_freq, d_sp_index);
}

/* Double precision. */
void oskar_scale_flux_by_spectral_index_cuda_d(int num_sources,
        double frequency, double* d_I, double* d_Q, double* d_U, double* d_V,
        double* d_ref_freq, const double* d_sp_index)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_sources + num_threads - 1) / num_threads;
    oskar_scale_flux_by_spectral_index_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_sources, frequency,
            d_I, d_Q, d_U, d_V, d_ref_freq, d_sp_index);
}

#ifdef __cplusplus
}
#endif


/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_scale_flux_by_spectral_index_cudak_f(const int num_sources,
        const float frequency, float* I, float* Q, float* U, float* V,
        float* ref_freq, const float* sp_index)
{
    int i;
    float scale;

    /* Get source index and check bounds. */
    i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_sources) return;

    /* Get scaling factor, multiply fluxes, update reference frequency. */
    scale = powf(frequency / ref_freq[i], sp_index[i]);
    I[i] *= scale;
    Q[i] *= scale;
    U[i] *= scale;
    V[i] *= scale;
    ref_freq[i] = frequency;
}

/* Double precision. */
__global__
void oskar_scale_flux_by_spectral_index_cudak_d(const int num_sources,
        const double frequency, double* I, double* Q, double* U, double* V,
        double* ref_freq, const double* sp_index)
{
    int i;
    double scale;

    /* Get source index and check bounds. */
    i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_sources) return;

    /* Get scaling factor, multiply fluxes, update reference frequency. */
    scale = pow(frequency / ref_freq[i], sp_index[i]);
    I[i] *= scale;
    Q[i] *= scale;
    U[i] *= scale;
    V[i] *= scale;
    ref_freq[i] = frequency;
}
