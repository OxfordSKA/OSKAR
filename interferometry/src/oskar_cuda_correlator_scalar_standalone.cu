/*
 * Copyright (c) 2011, The University of Oxford
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


#include "interferometry/oskar_cuda_correlator_scalar_standalone.h"

#include "interferometry/oskar_cuda_correlator_scalar.h"

int oskar_cudaf_correlator_scalar_standalone(const int num_antennas,
        const float * antenna_x, const float * antenna_y, const float * antenna_z,
        const int num_sources, const float * source_l, const float * source_m,
        const float * b_sqrt, const float * e, const float ra0, const float dec0,
        const float lst0, const int nsdt, const float sdt, const float k,
        const float lambda_bandwidth, float * vis)
{
    cudaError_t cuda_error = cudaSuccess;
    const int num_baselines = num_antennas * (num_antennas - 1) / 2;

    // Allocate device memory.
    // FIXME: perform allocations and memory copying in another function!
    float * d_antenna_x, * d_antenna_y, * d_antenna_z;
    float * d_source_l, * d_source_m, * d_source_n;
    float * d_eb;
    float * d_vis;
    float * d_work;

    cudaMalloc((void**)&d_antenna_x, num_antennas * sizeof(float));
    cudaMalloc((void**)&d_antenna_y, num_antennas * sizeof(float));
    cudaMalloc((void**)&d_antenna_z, num_antennas * sizeof(float));

    cudaMalloc((void**)&d_source_l, num_sources * sizeof(float));
    cudaMalloc((void**)&d_source_m, num_sources * sizeof(float));
    cudaMalloc((void**)&d_source_n, num_sources * sizeof(float));

    cudaMalloc((void**)&d_eb, num_sources * sizeof(float2));

    cudaMalloc((void**)&d_vis, num_baselines * sizeof(float2));

    size_t mem_work = num_antennas * (2 * num_sources + 3) * sizeof(float2);
    cudaMalloc((void**)&d_work, mem_work);

    // Copy memory to device.
    size_t mem_size = num_antennas * sizeof(float);
    cudaMemcpy(d_antenna_x, antenna_x, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_antenna_y, antenna_y, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_antenna_z, antenna_z, mem_size, cudaMemcpyHostToDevice);

    mem_size = num_sources * sizeof(float);
    float * source_n = (float*) malloc(mem_size);
    int i;
    // Evaluate n coordinate for each source.
    for (i = 0; i < num_sources; ++i)
    {
        float r2 = source_l[i] * source_l[i] + source_m[i] * source_m[i];
        source_n[i] = (r2 < 1.0) ? sqrtf(1.0f - r2) - 1.0f : -1.0f;
    }

    cudaMemcpy(d_source_l, source_l, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_m, source_m, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_n, source_n, mem_size, cudaMemcpyHostToDevice);


    // Scale source brightnesses by station beams (in E).
    mem_size = num_sources * num_antennas * sizeof(float2);
    float2 * eb = (float2*)malloc(mem_size);
    int j;
    for (j = 0; j < num_antennas; ++j)
    {
        for (i = 0; i < num_sources; ++i)
        {
            int idx = j * num_sources + i;
            eb[idx].x = e[2 * idx + 0] * b_sqrt[i];
            eb[idx].y = e[2 * idx + 1] * b_sqrt[i];
        }
    }
    cudaMemcpy(d_eb, eb, mem_size, cudaMemcpyHostToDevice);



    // Call the CUDA correlator function.
    oskar_cudaf_correlator_scalar(num_antennas, d_antenna_x, d_antenna_y,
            d_antenna_z, num_sources, d_source_l, d_source_m, d_source_n,
            d_eb, ra0, dec0, lst0, nsdt, sdt, lambda_bandwidth, d_vis, d_work);

    // Return memory to host.
    mem_size = num_baselines * sizeof(float2);
    cudaMemcpy(vis, d_vis, mem_size, cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(d_antenna_x);
    cudaFree(d_antenna_y);
    cudaFree(d_antenna_z);
    cudaFree(d_source_l);
    cudaFree(d_source_m);
    cudaFree(d_source_n);
    cudaFree(d_vis);
    cudaFree(d_work);

    return cuda_error;
}



int oskar_cudaf_correlator_scalar_allocate_memory(const unsigned num_antennas,
        const unsigned num_sources, float* d_antenna_x, float* d_antenna_y,
        float* d_antenna_z, float* d_source_l, float* d_source_m,
        float* d_source_n, float* d_eb, float* d_work)
{
    cudaError_t cuda_error = cudaSuccess;
    const unsigned num_baselines = num_antennas * (num_antennas - 1) / 2;

    cudaMalloc((void**)&d_antenna_x, num_antennas * sizeof(float));
    cudaMalloc((void**)&d_antenna_y, num_antennas * sizeof(float));
    cudaMalloc((void**)&d_antenna_z, num_antennas * sizeof(float));

    cudaMalloc((void**)&d_source_l, num_sources * sizeof(float));
    cudaMalloc((void**)&d_source_m, num_sources * sizeof(float));
    cudaMalloc((void**)&d_source_n, num_sources * sizeof(float));

    cudaMalloc((void**)&d_eb, num_sources * sizeof(float2));

    cudaMalloc((void**)&d_vis, num_baselines * sizeof(float2));

    size_t mem_work = num_antennas * (2 * num_sources + 3) * sizeof(float2);
    cudaMalloc((void**)&d_work, mem_work);

    // TODO check CUDA error code.

    return cuda_error;
}


