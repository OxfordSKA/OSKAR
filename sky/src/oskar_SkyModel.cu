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

#include "sky/oskar_SkyModel.h"
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_copy_gobal_sky_to_device_d(const oskar_SkyModelGlobal_d* h_sky,
        oskar_SkyModelGlobal_d* hd_sky)
{
    // Allocate memory for arrays in structure.
    size_t bytes        = h_sky->num_sources * sizeof(double);
    hd_sky->num_sources = h_sky->num_sources;

    cudaMalloc((void**)&hd_sky->RA,  bytes);
    cudaMalloc((void**)&hd_sky->Dec, bytes);
    cudaMalloc((void**)&hd_sky->I,   bytes);
    cudaMalloc((void**)&hd_sky->Q,   bytes);
    cudaMalloc((void**)&hd_sky->U,   bytes);
    cudaMalloc((void**)&hd_sky->V,   bytes);

    // Copy arrays to device.
    cudaMemcpy(hd_sky->RA,  h_sky->RA,  bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->Dec, h_sky->Dec, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->I,   h_sky->I,   bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->Q,   h_sky->Q,   bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->U,   h_sky->U,   bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->V,   h_sky->V,   bytes, cudaMemcpyHostToDevice);
}

void oskar_copy_gobal_sky_to_device_f(const oskar_SkyModelGlobal_f* h_sky,
        oskar_SkyModelGlobal_f* hd_sky)
{
    // Allocate memory for arrays in structure.
    size_t bytes        = h_sky->num_sources * sizeof(float);
    hd_sky->num_sources = h_sky->num_sources;

    cudaMalloc((void**)&hd_sky->RA,  bytes);
    cudaMalloc((void**)&hd_sky->Dec, bytes);
    cudaMalloc((void**)&hd_sky->I,   bytes);
    cudaMalloc((void**)&hd_sky->Q,   bytes);
    cudaMalloc((void**)&hd_sky->U,   bytes);
    cudaMalloc((void**)&hd_sky->V,   bytes);

    // Copy arrays to device.
    cudaMemcpy(hd_sky->RA,  h_sky->RA,  bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->Dec, h_sky->Dec, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->I,   h_sky->I,   bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->Q,   h_sky->Q,   bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->U,   h_sky->U,   bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->V,   h_sky->V,   bytes, cudaMemcpyHostToDevice);
}

void oskar_allocate_device_local_sky_d(const int num_sources,
        oskar_SkyModelLocal_d* hd_sky)
{
    size_t bytes = num_sources * sizeof(double);
    cudaMalloc((void**)&hd_sky->RA,    bytes);
    cudaMalloc((void**)&hd_sky->Dec,   bytes);
    cudaMalloc((void**)&hd_sky->I,     bytes);
    cudaMalloc((void**)&hd_sky->Q,     bytes);
    cudaMalloc((void**)&hd_sky->U,     bytes);
    cudaMalloc((void**)&hd_sky->V,     bytes);
    cudaMalloc((void**)&hd_sky->hor_l, bytes);
    cudaMalloc((void**)&hd_sky->hor_m, bytes);
    cudaMalloc((void**)&hd_sky->hor_n, bytes);
}

void oskar_allocate_device_local_sky_f(const int num_sources,
        oskar_SkyModelLocal_f* hd_sky)
{
    size_t bytes = num_sources * sizeof(float);
    cudaMalloc((void**)&hd_sky->RA,    bytes);
    cudaMalloc((void**)&hd_sky->Dec,   bytes);
    cudaMalloc((void**)&hd_sky->I,     bytes);
    cudaMalloc((void**)&hd_sky->Q,     bytes);
    cudaMalloc((void**)&hd_sky->U,     bytes);
    cudaMalloc((void**)&hd_sky->V,     bytes);
    cudaMalloc((void**)&hd_sky->hor_l, bytes);
    cudaMalloc((void**)&hd_sky->hor_m, bytes);
    cudaMalloc((void**)&hd_sky->hor_n, bytes);
}


void oskar_free_device_global_sky_d(oskar_SkyModelGlobal_d* hd_sky)
{
    cudaFree(hd_sky->RA);
    cudaFree(hd_sky->Dec);
    cudaFree(hd_sky->I);
    cudaFree(hd_sky->Q);
    cudaFree(hd_sky->U);
    cudaFree(hd_sky->V);
}


void oskar_free_device_global_sky_f(oskar_SkyModelGlobal_f* hd_sky)
{
    cudaFree(hd_sky->RA);
    cudaFree(hd_sky->Dec);
    cudaFree(hd_sky->I);
    cudaFree(hd_sky->Q);
    cudaFree(hd_sky->U);
    cudaFree(hd_sky->V);
}

void oskar_free_device_local_sky_d(oskar_SkyModelLocal_d* hd_sky)
{
    cudaFree(hd_sky->RA);
    cudaFree(hd_sky->Dec);
    cudaFree(hd_sky->I);
    cudaFree(hd_sky->Q);
    cudaFree(hd_sky->U);
    cudaFree(hd_sky->V);
    cudaFree(hd_sky->hor_l);
    cudaFree(hd_sky->hor_m);
    cudaFree(hd_sky->hor_n);
}

void oskar_free_device_local_sky_f(oskar_SkyModelLocal_f* hd_sky)
{
    cudaFree(hd_sky->RA);
    cudaFree(hd_sky->Dec);
    cudaFree(hd_sky->I);
    cudaFree(hd_sky->Q);
    cudaFree(hd_sky->U);
    cudaFree(hd_sky->V);
    cudaFree(hd_sky->hor_l);
    cudaFree(hd_sky->hor_m);
    cudaFree(hd_sky->hor_n);
}

#ifdef __cplusplus
}
#endif

