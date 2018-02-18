/*
 * Copyright (c) 2018, The University of Oxford
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

#include "math/oskar_fftphase_cuda.h"
#include <vector_types.h>

template <typename FP>
__global__
void oskar_fftphase_cudak(const int num_x, const int num_y, FP* complex_data)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= num_x || iy >= num_y) return;
    const int x = 1 - (((ix + iy) & 1) << 1);
    complex_data[((iy * num_x + ix) << 1)]     *= x;
    complex_data[((iy * num_x + ix) << 1) + 1] *= x;
}

#ifdef __cplusplus
extern "C" {
#endif

void oskar_fftphase_cuda_cf(const int num_x, const int num_y,
        float* d_complex_data)
{
    dim3 num_blocks, num_threads(16, 16, 1);
    num_blocks.x = (num_x + num_threads.x - 1) / num_threads.x;
    num_blocks.y = (num_y + num_threads.y - 1) / num_threads.y;
    oskar_fftphase_cudak OSKAR_CUDAK_CONF(num_blocks, num_threads) (
            num_x, num_y, d_complex_data);
}

void oskar_fftphase_cuda_cd(const int num_x, const int num_y,
        double* d_complex_data)
{
    dim3 num_blocks, num_threads(16, 16, 1);
    num_blocks.x = (num_x + num_threads.x - 1) / num_threads.x;
    num_blocks.y = (num_y + num_threads.y - 1) / num_threads.y;
    oskar_fftphase_cudak OSKAR_CUDAK_CONF(num_blocks, num_threads) (
            num_x, num_y, d_complex_data);
}

#ifdef __cplusplus
}
#endif
