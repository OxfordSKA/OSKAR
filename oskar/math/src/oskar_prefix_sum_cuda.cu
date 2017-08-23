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

#include "math/oskar_prefix_sum_cuda.h"

template <typename T>
__global__ void oskar_prefix_sum_cudak(const int num_elements,
        const T* in, T* out, T* block_sums, const T init_val,
        const int exclusive, const int block_size)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    int pos = threadIdx.x;

    // Copy input data to local memory.
    extern __shared__ T scratch[];
    scratch[pos] = 0;
    pos += block_size;
    scratch[pos] = 0;
    if (i < num_elements)
    {
        if (exclusive)
        {
            const T local_init = (i == 0) ? init_val : 0;
            scratch[pos] = (threadIdx.x > 0) ? in[i - 1] : local_init;
        }
        else
        {
            scratch[pos] = in[i];
        }
    }

    // Prefix sum.
    for (int j = 1; j < block_size; j <<= 1)
    {
        __syncthreads();
        const T x = scratch[pos - j];
        __syncthreads();
        scratch[pos] += x;
    }

    // Store local results.
    if (i < num_elements) out[i] = scratch[pos];

    // Store sum for the block.
    if (threadIdx.x == block_size - 1)
    {
        const T x = (i < num_elements) ? in[i] : 0;
        block_sums[blockIdx.x] = exclusive ?
                x + scratch[pos] : scratch[pos];
    }
}

template <typename T>
__global__ void oskar_prefix_sum_finalise_cudak(const int num_elements,
        T* out, const T* block_sums, int offset)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x + offset;
    if (i < num_elements) out[i] += block_sums[blockIdx.x];
}

#ifdef __cplusplus
extern "C" {
#endif

void oskar_prefix_sum_cuda_int(int num_elements, const int* d_in, int* d_out,
        int num_blocks, int block_size, int* d_block_sums, int init_val,
        int exclusive)
{
    int shared_mem = 2 * block_size * sizeof(int);
    oskar_prefix_sum_cudak
    OSKAR_CUDAK_CONF(num_blocks, block_size, shared_mem) (
            num_elements, d_in, d_out, d_block_sums, init_val,
            exclusive, block_size);
}

void oskar_prefix_sum_finalise_cuda_int(int num_elements, int* d_out,
        int num_blocks, int block_size, const int* d_block_sums, int offset)
{
    oskar_prefix_sum_finalise_cudak OSKAR_CUDAK_CONF(num_blocks, block_size) (
            num_elements, d_out, d_block_sums, offset);
}

#ifdef __cplusplus
}
#endif
