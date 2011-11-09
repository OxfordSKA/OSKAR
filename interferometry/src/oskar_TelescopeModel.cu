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
#include "interferometry/oskar_telescope_model_copy.h"
#include "interferometry/oskar_telescope_model_free.h"
#include "interferometry/oskar_telescope_model_init.h"
#include "interferometry/oskar_telescope_model_load_station_pos.h"
#include "interferometry/oskar_telescope_model_location.h"
#include "interferometry/oskar_telescope_model_multiply_by_wavenumber.h"
#include "interferometry/oskar_telescope_model_type.h"
#include "station/oskar_station_model_load.h"
#include "math/cudak/oskar_cudak_vec_scale_rr.h" // DEPRECATED
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cmath>

oskar_TelescopeModel::oskar_TelescopeModel(int type, int location,
        int n_stations)
: num_stations(0),
  station(NULL)
{
    if (oskar_telescope_model_init(this, type, location, n_stations))
        throw "Error in oskar_telescope_model_init.";
}

oskar_TelescopeModel::oskar_TelescopeModel(const oskar_TelescopeModel* other,
        int location)
: num_stations(0),
  station(NULL)
{
    if (oskar_telescope_model_init(this, other->station_x.type(), location,
            other->num_stations))
        throw "Error in oskar_telescope_model_init.";
    if (oskar_telescope_model_copy(this, other)) // Copy other to this.
        throw "Error in oskar_telescope_model_copy.";
}

oskar_TelescopeModel::~oskar_TelescopeModel()
{
    if (oskar_telescope_model_free(this))
        throw "Error in oskar_telescope_model_free.";
}

int oskar_TelescopeModel::load_station_pos(const char* filename,
        double longitude, double latitude, double altitude)
{
    return oskar_telescope_model_load_station_pos(this, filename,
            longitude, latitude, altitude);
}

int oskar_TelescopeModel::location() const
{
    return oskar_telescope_model_location(this);
}

int oskar_TelescopeModel::load_station(int index, const char* filename)
{
    if (index >= this->num_stations)
        return OSKAR_ERR_OUT_OF_RANGE;
    return oskar_station_model_load(&(this->station[index]), filename);
}

int oskar_TelescopeModel::multiply_by_wavenumber(double frequency_hz)
{
    return oskar_telescope_model_multiply_by_wavenumber(this, frequency_hz);
}

int oskar_TelescopeModel::type() const
{
    return oskar_telescope_model_type(this);
}


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
