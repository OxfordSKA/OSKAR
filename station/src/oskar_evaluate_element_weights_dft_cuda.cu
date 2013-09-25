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

#include <oskar_evaluate_element_weights_dft_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_evaluate_element_weights_dft_cuda_f(float2* d_weights,
        int num_elements, float wavenumber, const float* d_x,
        const float* d_y, const float* d_z, float x_beam, float y_beam,
        float z_beam)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_elements + num_threads - 1) / num_threads;
    oskar_evaluate_element_weights_dft_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (d_weights, num_elements,
            wavenumber, d_x, d_y, d_z, x_beam, y_beam, z_beam);
}

/* Double precision. */
void oskar_evaluate_element_weights_dft_cuda_d(double2* d_weights,
        int num_elements, double wavenumber, const double* d_x,
        const double* d_y, const double* d_z, double x_beam, double y_beam,
        double z_beam)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_elements + num_threads - 1) / num_threads;
    oskar_evaluate_element_weights_dft_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (d_weights, num_elements,
            wavenumber, d_x, d_y, d_z, x_beam, y_beam, z_beam);
}


/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_evaluate_element_weights_dft_cudak_f(float2* weights,
        const int n_in, const float wavenumber, const float* x_in,
        const float* y_in, const float* z_in, const float x_out,
        const float y_out, const float z_out)
{
    float cxi, cyi, czi, phase;
    float2 weight;

    /* Get input index. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_in) return;

    /* Cache input data from global memory. */
    cxi = wavenumber * x_in[i];
    cyi = wavenumber * y_in[i];
    czi = wavenumber * z_in[i];

    /* Compute the geometric phase of the output direction. */
    phase =  cxi * x_out;
    phase += cyi * y_out;
    phase += czi * z_out;
    sincosf(-phase, &weight.y, &weight.x);

    /* Write result to global memory. */
    weights[i] = weight;
}

/* Double precision. */
__global__
void oskar_evaluate_element_weights_dft_cudak_d(double2* weights,
        const int n_in, const double wavenumber, const double* x_in,
        const double* y_in, const double* z_in, const double x_out,
        const double y_out, const double z_out)
{
    double cxi, cyi, czi, phase;
    double2 weight;

    /* Get input index. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_in) return;

    /* Cache input data from global memory. */
    cxi = wavenumber * x_in[i];
    cyi = wavenumber * y_in[i];
    czi = wavenumber * z_in[i];

    /* Compute the geometric phase of the output direction. */
    phase =  cxi * x_out;
    phase += cyi * y_out;
    phase += czi * z_out;
    sincos(-phase, &weight.y, &weight.x);

    /* Write result to global memory. */
    weights[i] = weight;
}

#ifdef __cplusplus
}
#endif
