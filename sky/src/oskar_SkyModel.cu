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
#include <cstdio>


oskar_SkyModel::oskar_SkyModel(const int num_sources, const int type, const int location)
: private_num_sources(num_sources),
  RA(type, location, num_sources),
  Dec(type, location, num_sources),
  I(type, location, num_sources),
  Q(type, location, num_sources),
  U(type, location, num_sources),
  V(type, location, num_sources),
  reference_freq(type, location, num_sources),
  spectral_index(type, location, num_sources),
  update_timestamp(0.0),
  rel_l(type, location, num_sources),
  rel_m(type, location, num_sources),
  rel_n(type, location, num_sources),
  hor_l(type, location, num_sources),
  hor_m(type, location, num_sources),
  hor_n(type, location, num_sources)
{
}


oskar_SkyModel::oskar_SkyModel(const oskar_SkyModel* sky, const int location)
: private_num_sources(sky->num_sources()),
  RA(&sky->RA, location),
  Dec(&sky->Dec, location),
  I(&sky->I, location),
  Q(&sky->Q, location),
  U(&sky->U, location),
  V(&sky->V, location),
  reference_freq(&sky->reference_freq, location),
  spectral_index(&sky->spectral_index, location),
  update_timestamp(sky->update_timestamp),
  rel_l(&sky->rel_l, location),
  rel_m(&sky->rel_m, location),
  rel_n(&sky->rel_n, location),
  hor_l(&sky->rel_l, location),
  hor_m(&sky->rel_m, location),
  hor_n(&sky->rel_n, location)
{
}

oskar_SkyModel::oskar_SkyModel(const char* filename, const int type, const int location)
: private_num_sources(0),
  RA(type, location, 0),
  Dec(type, location, 0),
  I(type, location, 0),
  Q(type, location, 0),
  U(type, location, 0),
  V(type, location, 0),
  reference_freq(type, location, 0),
  spectral_index(type, location, 0),
  update_timestamp(0.0),
  rel_l(type, location, 0),
  rel_m(type, location, 0),
  rel_n(type, location, 0),
  hor_l(type, location, 0),
  hor_m(type, location, 0),
  hor_n(type, location, 0)
{
}

oskar_SkyModel::~oskar_SkyModel()
{

}


int oskar_SkyModel::load(const char* filename, const int type, const int location)
{

    return 0;
}







// ========== DEPRECATED ======================================================
#ifdef __cplusplus
extern "C" {
#endif
void oskar_sky_model_global_copy_to_gpu_d(const oskar_SkyModelGlobal_d* h_sky,
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
    cudaMalloc((void**)&hd_sky->reference_freq, bytes);
    cudaMalloc((void**)&hd_sky->spectral_index, bytes);
    cudaMalloc((void**)&hd_sky->rel_l, bytes);
    cudaMalloc((void**)&hd_sky->rel_m, bytes);
    cudaMalloc((void**)&hd_sky->rel_n, bytes);


    // Copy arrays to device.
    cudaMemcpy(hd_sky->RA,  h_sky->RA,  bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->Dec, h_sky->Dec, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->I,   h_sky->I,   bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->Q,   h_sky->Q,   bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->U,   h_sky->U,   bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->V,   h_sky->V,   bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->reference_freq, h_sky->reference_freq, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->spectral_index, h_sky->spectral_index, bytes, cudaMemcpyHostToDevice);
}

void oskar_sky_model_global_copy_to_gpu_f(const oskar_SkyModelGlobal_f* h_sky,
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
    cudaMalloc((void**)&hd_sky->reference_freq, bytes);
    cudaMalloc((void**)&hd_sky->spectral_index, bytes);
    cudaMalloc((void**)&hd_sky->rel_l, bytes);
    cudaMalloc((void**)&hd_sky->rel_m, bytes);
    cudaMalloc((void**)&hd_sky->rel_n, bytes);

    // Copy arrays to device.
    cudaMemcpy(hd_sky->RA,  h_sky->RA,  bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->Dec, h_sky->Dec, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->I,   h_sky->I,   bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->Q,   h_sky->Q,   bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->U,   h_sky->U,   bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->V,   h_sky->V,   bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->reference_freq, h_sky->reference_freq, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->spectral_index, h_sky->spectral_index, bytes, cudaMemcpyHostToDevice);
}

void oskar_local_sky_model_allocate_gpu_d(const int num_sources,
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
    cudaMalloc((void**)&hd_sky->rel_l, bytes);
    cudaMalloc((void**)&hd_sky->rel_m, bytes);
    cudaMalloc((void**)&hd_sky->rel_n, bytes);
}

void oskar_local_sky_model_allocate_gpu_f(const int num_sources,
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
    cudaMalloc((void**)&hd_sky->rel_l, bytes);
    cudaMalloc((void**)&hd_sky->rel_m, bytes);
    cudaMalloc((void**)&hd_sky->rel_n, bytes);
}


void oskar_global_sky_model_free_gpu_d(oskar_SkyModelGlobal_d* hd_sky)
{
    cudaFree(hd_sky->RA);
    cudaFree(hd_sky->Dec);
    cudaFree(hd_sky->I);
    cudaFree(hd_sky->Q);
    cudaFree(hd_sky->U);
    cudaFree(hd_sky->V);
    cudaFree(hd_sky->reference_freq);
    cudaFree(hd_sky->spectral_index);
    cudaFree(hd_sky->rel_l);
    cudaFree(hd_sky->rel_m);
    cudaFree(hd_sky->rel_n);
}


void oskar_global_sky_model_free_gpu_f(oskar_SkyModelGlobal_f* hd_sky)
{
    cudaFree(hd_sky->RA);
    cudaFree(hd_sky->Dec);
    cudaFree(hd_sky->I);
    cudaFree(hd_sky->Q);
    cudaFree(hd_sky->U);
    cudaFree(hd_sky->V);
    cudaFree(hd_sky->reference_freq);
    cudaFree(hd_sky->spectral_index);
    cudaFree(hd_sky->rel_l);
    cudaFree(hd_sky->rel_m);
    cudaFree(hd_sky->rel_n);
}

void oskar_local_sky_model_free_gpu_d(oskar_SkyModelLocal_d* hd_sky)
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
    cudaFree(hd_sky->rel_l);
    cudaFree(hd_sky->rel_m);
    cudaFree(hd_sky->rel_n);
}

void oskar_local_sky_model_free_gpu_f(oskar_SkyModelLocal_f* hd_sky)
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
    cudaFree(hd_sky->rel_l);
    cudaFree(hd_sky->rel_m);
    cudaFree(hd_sky->rel_n);
}

#ifdef __cplusplus
}
#endif

