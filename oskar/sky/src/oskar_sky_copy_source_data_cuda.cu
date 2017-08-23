/*
 * Copyright (c) 2017, The University of Oxford
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

#include "sky/oskar_sky_copy_source_data_cuda.h"
#include <cuda_runtime_api.h>

template<typename T>
__global__
void oskar_sky_copy_source_data_cudak(const int num,
        const int* restrict mask, const int* restrict indices,
        const T* ra_in,   T* ra_out,
        const T* dec_in,  T* dec_out,
        const T* I_in,    T* I_out,
        const T* Q_in,    T* Q_out,
        const T* U_in,    T* U_out,
        const T* V_in,    T* V_out,
        const T* ref_in,  T* ref_out,
        const T* sp_in,   T* sp_out,
        const T* rm_in,   T* rm_out,
        const T* l_in,    T* l_out,
        const T* m_in,    T* m_out,
        const T* n_in,    T* n_out,
        const T* a_in,    T* a_out,
        const T* b_in,    T* b_out,
        const T* c_in,    T* c_out,
        const T* maj_in,  T* maj_out,
        const T* min_in,  T* min_out,
        const T* pa_in,   T* pa_out
        )
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num) return;
    if (mask[i])
    {
        int i_out = indices[i];
        ra_out[i_out]  = ra_in[i];
        dec_out[i_out] = dec_in[i];
        I_out[i_out]   = I_in[i];
        Q_out[i_out]   = Q_in[i];
        U_out[i_out]   = U_in[i];
        V_out[i_out]   = V_in[i];
        ref_out[i_out] = ref_in[i];
        sp_out[i_out]  = sp_in[i];
        rm_out[i_out]  = rm_in[i];
        l_out[i_out]   = l_in[i];
        m_out[i_out]   = m_in[i];
        n_out[i_out]   = n_in[i];
        a_out[i_out]   = a_in[i];
        b_out[i_out]   = b_in[i];
        c_out[i_out]   = c_in[i];
        maj_out[i_out] = maj_in[i];
        min_out[i_out] = min_in[i];
        pa_out[i_out]  = pa_in[i];
    }
}

void oskar_sky_copy_source_data_cuda_f(int num,
        int* num_out, const int* mask, const int* indices,
        const float* ra_in,   float* ra_out,
        const float* dec_in,  float* dec_out,
        const float* I_in,    float* I_out,
        const float* Q_in,    float* Q_out,
        const float* U_in,    float* U_out,
        const float* V_in,    float* V_out,
        const float* ref_in,  float* ref_out,
        const float* sp_in,   float* sp_out,
        const float* rm_in,   float* rm_out,
        const float* l_in,    float* l_out,
        const float* m_in,    float* m_out,
        const float* n_in,    float* n_out,
        const float* a_in,    float* a_out,
        const float* b_in,    float* b_out,
        const float* c_in,    float* c_out,
        const float* maj_in,  float* maj_out,
        const float* min_in,  float* min_out,
        const float* pa_in,   float* pa_out
        )
{
    int num_blocks, num_threads = 256;
    num_blocks = (num + num_threads - 1) / num_threads;
    oskar_sky_copy_source_data_cudak<float>
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (
            num, mask, indices, ra_in, ra_out, dec_in, dec_out,
            I_in, I_out, Q_in, Q_out, U_in, U_out, V_in, V_out,
            ref_in, ref_out, sp_in, sp_out, rm_in, rm_out,
            l_in, l_out, m_in, m_out, n_in, n_out,
            a_in, a_out, b_in, b_out, c_in, c_out,
            maj_in, maj_out, min_in, min_out, pa_in, pa_out);
    cudaMemcpy(num_out, &indices[num-1], sizeof(int), cudaMemcpyDeviceToHost);
    (*num_out)++;
}

void oskar_sky_copy_source_data_cuda_d(int num,
        int* num_out, const int* mask, const int* indices,
        const double* ra_in,   double* ra_out,
        const double* dec_in,  double* dec_out,
        const double* I_in,    double* I_out,
        const double* Q_in,    double* Q_out,
        const double* U_in,    double* U_out,
        const double* V_in,    double* V_out,
        const double* ref_in,  double* ref_out,
        const double* sp_in,   double* sp_out,
        const double* rm_in,   double* rm_out,
        const double* l_in,    double* l_out,
        const double* m_in,    double* m_out,
        const double* n_in,    double* n_out,
        const double* a_in,    double* a_out,
        const double* b_in,    double* b_out,
        const double* c_in,    double* c_out,
        const double* maj_in,  double* maj_out,
        const double* min_in,  double* min_out,
        const double* pa_in,   double* pa_out
        )
{
    int num_blocks, num_threads = 256;
    num_blocks = (num + num_threads - 1) / num_threads;
    oskar_sky_copy_source_data_cudak<double>
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (
            num, mask, indices, ra_in, ra_out, dec_in, dec_out,
            I_in, I_out, Q_in, Q_out, U_in, U_out, V_in, V_out,
            ref_in, ref_out, sp_in, sp_out, rm_in, rm_out,
            l_in, l_out, m_in, m_out, n_in, n_out,
            a_in, a_out, b_in, b_out, c_in, c_out,
            maj_in, maj_out, min_in, min_out, pa_in, pa_out);
    cudaMemcpy(num_out, &indices[num-1], sizeof(int), cudaMemcpyDeviceToHost);
    (*num_out)++;
}
