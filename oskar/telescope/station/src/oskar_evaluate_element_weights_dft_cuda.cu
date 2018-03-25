/*
 * Copyright (c) 2012-2018, The University of Oxford
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

#include "telescope/station/oskar_evaluate_element_weights_dft_cuda.h"

template <typename FP, typename FP2>
__global__
void oskar_evaluate_element_weights_dft_cudak(const int num_elements,
        const FP* restrict x, const FP* restrict y, const FP* restrict z,
        const FP wavenumber, const FP x1, const FP y1, const FP z1,
        FP2* restrict weights)
{
    FP2 weight;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_elements) return;
    const FP p = wavenumber * (x[i] * x1 + y[i] * y1 + z[i] * z1);
    sincos((FP)(-p), &weight.y, &weight.x);
    weights[i] = weight;
}

extern "C" {

void oskar_evaluate_element_weights_dft_cuda_f(int num_elements,
        const float* d_x, const float* d_y, const float* d_z,
        float wavenumber, float x_beam, float y_beam, float z_beam,
        float2* d_weights)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_elements + num_threads - 1) / num_threads;
    oskar_evaluate_element_weights_dft_cudak<float, float2>
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_elements,
            d_x, d_y, d_z, wavenumber, x_beam, y_beam, z_beam, d_weights);
}

void oskar_evaluate_element_weights_dft_cuda_d(int num_elements,
        const double* d_x, const double* d_y, const double* d_z,
        double wavenumber, double x_beam, double y_beam, double z_beam,
        double2* d_weights)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_elements + num_threads - 1) / num_threads;
    oskar_evaluate_element_weights_dft_cudak<double, double2>
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_elements,
            d_x, d_y, d_z, wavenumber, x_beam, y_beam, z_beam, d_weights);
}

}
