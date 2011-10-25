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


#include "interferometry/oskar_TelescopeModel.h"
#include "math/cudak/oskar_cudak_vec_scale_rr.h"
#include <cuda_runtime_api.h>













#ifdef __cplusplus
extern "C" {
#endif
// DEPRECATED
void oskar_copy_telescope_to_device_d(const oskar_TelescopeModel_d* h_telescope,
        oskar_TelescopeModel_d* hd_telescope)
{
    size_t mem_size = h_telescope->num_antennas * sizeof(double);

    hd_telescope->num_antennas = h_telescope->num_antennas;

    cudaMalloc((void**)&(hd_telescope->antenna_x), mem_size);
    cudaMalloc((void**)&(hd_telescope->antenna_y), mem_size);
    cudaMalloc((void**)&(hd_telescope->antenna_z), mem_size);

    cudaMemcpy(hd_telescope->antenna_x, h_telescope->antenna_x, mem_size,
            cudaMemcpyHostToDevice);
    cudaMemcpy(hd_telescope->antenna_y, h_telescope->antenna_y, mem_size,
            cudaMemcpyHostToDevice);
    cudaMemcpy(hd_telescope->antenna_z, h_telescope->antenna_z, mem_size,
            cudaMemcpyHostToDevice);
}

// DEPRECATED
void oskar_copy_telescope_to_device_f(const oskar_TelescopeModel_f* h_telescope,
        oskar_TelescopeModel_f* hd_telescope)
{
    size_t mem_size = h_telescope->num_antennas * sizeof(float);

    hd_telescope->num_antennas = h_telescope->num_antennas;

    cudaMalloc((void**)&(hd_telescope->antenna_x), mem_size);
    cudaMalloc((void**)&(hd_telescope->antenna_y), mem_size);
    cudaMalloc((void**)&(hd_telescope->antenna_z), mem_size);

    cudaMemcpy(hd_telescope->antenna_x, h_telescope->antenna_x, mem_size,
            cudaMemcpyHostToDevice);
    cudaMemcpy(hd_telescope->antenna_y, h_telescope->antenna_y, mem_size,
            cudaMemcpyHostToDevice);
    cudaMemcpy(hd_telescope->antenna_z, h_telescope->antenna_z, mem_size,
            cudaMemcpyHostToDevice);
}

// DEPRECATED
void oskar_scale_device_telescope_coords_d(oskar_TelescopeModel_d* hd_telescope,
        const double value)
{
    int num_stations = hd_telescope->num_antennas;
    int num_threads  = 256;
    int num_blocks   = (int)ceil((double) num_stations / num_threads);
    oskar_cudak_vec_scale_rr_d <<< num_blocks, num_threads >>>
            (num_stations, value, hd_telescope->antenna_x);
    oskar_cudak_vec_scale_rr_d <<< num_blocks, num_threads >>>
            (num_stations, value, hd_telescope->antenna_y);
    oskar_cudak_vec_scale_rr_d <<< num_blocks, num_threads >>>
            (num_stations, value, hd_telescope->antenna_z);
}

// DEPRECATED
void oskar_scale_device_telescope_coords_f(oskar_TelescopeModel_f* hd_telescope,
        const float value)
{
    int num_stations = hd_telescope->num_antennas;
    int num_threads  = 256;
    int num_blocks   = (int)ceilf((float) num_stations / num_threads);
    oskar_cudak_vec_scale_rr_f <<< num_blocks, num_threads >>>
            (num_stations, value, hd_telescope->antenna_x);
    oskar_cudak_vec_scale_rr_f <<< num_blocks, num_threads >>>
            (num_stations, value, hd_telescope->antenna_y);
    oskar_cudak_vec_scale_rr_f <<< num_blocks, num_threads >>>
            (num_stations, value, hd_telescope->antenna_z);
}


void oskar_free_device_telescope_d(oskar_TelescopeModel_d* hd_telescope)
{
    cudaFree(hd_telescope->antenna_x);
    cudaFree(hd_telescope->antenna_y);
    cudaFree(hd_telescope->antenna_z);
}

void oskar_free_device_telescope_f(oskar_TelescopeModel_f* hd_telescope)
{
    cudaFree(hd_telescope->antenna_x);
    cudaFree(hd_telescope->antenna_y);
    cudaFree(hd_telescope->antenna_z);
}



#ifdef __cplusplus
}
#endif
