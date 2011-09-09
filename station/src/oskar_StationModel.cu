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


#include "station/oskar_StationModel.h"
#include "math/cudak/oskar_cudak_vec_scale_rr.h"
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_copy_stations_to_device_d(const oskar_StationModel_d* h_stations,
        const unsigned num_stations, oskar_StationModel_d* hd_stations)
{
    // Allocate and copy memory for each station.
    for (unsigned i = 0; i < num_stations; ++i)
    {
        hd_stations[i].num_antennas = h_stations[i].num_antennas;
        size_t mem_size = hd_stations[i].num_antennas * sizeof(double);
        cudaMalloc((void**)&(hd_stations[i].antenna_x), mem_size);
        cudaMalloc((void**)&(hd_stations[i].antenna_y), mem_size);
        cudaMemcpy(hd_stations[i].antenna_x, h_stations[i].antenna_x, mem_size,
                cudaMemcpyHostToDevice);
        cudaMemcpy(hd_stations[i].antenna_y, h_stations[i].antenna_y, mem_size,
                cudaMemcpyHostToDevice);
    }
}


void oskar_copy_stations_to_device_f(const oskar_StationModel_f* h_stations,
        const unsigned num_stations, oskar_StationModel_f* hd_stations)
{
    // Allocate and copy memory for each station.
    for (unsigned i = 0; i < num_stations; ++i)
    {
        hd_stations[i].num_antennas = h_stations[i].num_antennas;
        size_t mem_size = hd_stations[i].num_antennas * sizeof(float);
        cudaMalloc((void**)&(hd_stations[i].antenna_x), mem_size);
        cudaMalloc((void**)&(hd_stations[i].antenna_y), mem_size);
        cudaMemcpy(hd_stations[i].antenna_x, h_stations[i].antenna_x, mem_size,
                cudaMemcpyHostToDevice);
        cudaMemcpy(hd_stations[i].antenna_y, h_stations[i].antenna_y, mem_size,
                cudaMemcpyHostToDevice);
    }
}

void oskar_scale_station_coords_d(const unsigned num_stations,
        oskar_StationModel_d* hd_stations, const double value)
{
    int num_threads = 256;
    for (unsigned i = 0; i < num_stations; ++i)
    {
        int num_antennas = hd_stations[i].num_antennas;
        int num_blocks  = (int)ceil((double) num_antennas / num_threads);
        oskar_cudak_vec_scale_rr_d <<< num_blocks, num_threads >>>
                (num_antennas, value, hd_stations[i].antenna_x);
        oskar_cudak_vec_scale_rr_d <<< num_blocks, num_threads >>>
                (num_antennas, value, hd_stations[i].antenna_y);
    }
}


void oskar_scale_station_coords_f(const unsigned num_stations,
        oskar_StationModel_f* hd_stations, const float value)
{
    int num_threads = 256;
    for (unsigned i = 0; i < num_stations; ++i)
    {
        int num_antennas = hd_stations[i].num_antennas;
        int num_blocks  = (int)ceil((float) num_antennas / num_threads);
        oskar_cudak_vec_scale_rr_f <<< num_blocks, num_threads >>>
                (num_antennas, value, hd_stations[i].antenna_x);
        oskar_cudak_vec_scale_rr_f <<< num_blocks, num_threads >>>
                (num_antennas, value, hd_stations[i].antenna_y);
    }
}


#ifdef __cplusplus
}
#endif
