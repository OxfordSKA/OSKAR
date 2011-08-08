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

#include "interferometry/oskar_interferometer1_scalar.h"
#include "interferometry/oskar_cuda_correlator_scalar.h"
#include "interferometry/cudak/oskar_cudak_correlator.h"
#include "interferometry/cudak/oskar_cudak_xyz_to_uvw.h"

#include "math/cudak/oskar_cudak_dftw_3d_seq_out.h"
#include "math/cudak/oskar_cudak_mat_mul_cc.h"

#include "sky/oskar_mjd_to_last_fast.h"


#include <cstdio>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SEC_PER_DAY 86400.0

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
void oskar_cuda_copy_telescope_to_gpu_d(const oskar_TelescopeModel* h_telescope,
        oskar_TelescopeModel* hd_telescope);

void oskar_cuda_copy_stations_to_gpu_d(const oskar_StationModel* h_stations,
        const unsigned num_stations, oskar_StationModel* hd_stations);

void oskar_cuda_copy_global_sky_to_gpu_d(const oskar_SkyModelGlobal_d* h_sky,
        oskar_SkyModelGlobal_d* d_sky);

void oskar_cuda_alloc_local_sky_d(int num_sources, oskar_SkyModelLocal_d* d_sky);
//------------------------------------------------------------------------------



int oskar_interferometer1_scalar_d(
        const oskar_TelescopeModel telescope, // NOTE: In ITRS coordinates
        const oskar_StationModel * stations,
        const oskar_SkyModelGlobal_d sky,
        const double ra0_rad,
        const double dec0_rad,
        const double start_mjd_utc,
        const double obs_length_days,
        const unsigned n_vis_dumps,
        const unsigned n_vis_ave,
        const unsigned n_fringe_ave,
        const double freq,
        const double bandwidth,
        double2 * vis
){
    cudaError_t cuda_error = cudaSuccess;

    // === Evaluate number of stations and number of baselines.
    const unsigned num_stations = telescope.num_antennas;
    const unsigned num_baselines = num_stations * (num_stations - 1) / 2;

    // === Allocate device memory for telescope and transfer to device.
    oskar_TelescopeModel hd_telescope;
    oskar_cuda_copy_telescope_to_gpu_d(&telescope, &hd_telescope);

    // === Allocate device memory for antennas and transfer to the device.
    size_t mem_size = num_stations * sizeof(oskar_StationModel);
    oskar_StationModel * hd_stations = (oskar_StationModel*)malloc(mem_size);
    oskar_cuda_copy_stations_to_gpu_d(stations, num_stations, hd_stations);

    // === Allocate device memory for source model and transfer to device.
    oskar_SkyModelGlobal_d d_sky_global;
    oskar_cuda_copy_global_sky_to_gpu_d(&sky, &d_sky_global);

    // Allocate local sky structure.
    oskar_SkyModelLocal_d d_sky_local;
    oskar_cuda_alloc_local_sky_d(sky.num_sources, &d_sky_local);

    // Calculate time increments.
    unsigned total_samples = n_vis_dumps * n_fringe_ave * n_vis_ave;
    double dt_days = obs_length_days / total_samples;
    double dt_vis_days = obs_length_days / n_vis_dumps;
    double dt_vis_ave_days = dt_vis_days / n_vis_ave;
    double dt_vis_offset = dt_vis_days / 2.0;
    double dt_vis_ave_offset = dt_vis_ave_days / 2.0;

    // Loop over number of visibility snapshots.
    for (int j = 0; j < n_vis_dumps; ++j)
    {
        // Start time for the visibility dump.
        double t_vis_start = start_mjd_utc + j * dt_vis_days;

        // Loop over evaluations of the visibility average with changing
        // E-Jones within the dump.
        for (int i = 0; i < n_vis_ave; ++i)
        {
            double t_ave_start = t_vis_start + i * dt_vis_ave_days;
            double t_ave_mid = t_ave_start + dt_vis_ave_offset;
            double lst = oskar_mjd_to_last_fast_d(t_ave_mid,
                    telescope.longitude);

            // 6. Find sources above horizon.
            // TODO

            // 7. Evaluate E-Jones for each source position per station.
            // TODO: optimisation if all stations are the same?
            //for each station
            // oskar_cudad_bp2hc() <=== move this to beamforming folder.

            // 8. Correlator which updates phase matrix.
            // TODO
            // oskar_cudad_correlator_sclar()

            // 9. Accumulate visibilities.
            // TODO
        }

        // 10. Dump a new set of visibilities including baseline coordinates.
        // TODO
    }


    return (int)cuda_error;
}




void oskar_cuda_copy_telescope_to_gpu_d(const oskar_TelescopeModel * h_telescope,
        oskar_TelescopeModel * hd_telescope)
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


void oskar_cuda_copy_stations_to_gpu_d(const oskar_StationModel * h_stations,
        const unsigned num_stations, oskar_StationModel * hd_stations)
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


void oskar_cuda_copy_global_sky_to_gpu_d(const oskar_SkyModelGlobal_d* h_sky,
        oskar_SkyModelGlobal_d* d_sky)
{
    // Allocate memory for arrays in structure.
    size_t bytes = h_sky->num_sources * sizeof(double);
    cudaMalloc((void**)&(d_sky->RA), bytes);
    cudaMalloc((void**)&(d_sky->Dec), bytes);
    cudaMalloc((void**)&(d_sky->I), bytes);
    cudaMalloc((void**)&(d_sky->Q), bytes);
    cudaMalloc((void**)&(d_sky->U), bytes);
    cudaMalloc((void**)&(d_sky->V), bytes);

    // Copy arrays to device.
    cudaMemcpy(d_sky->RA, h_sky->RA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sky->Dec, h_sky->Dec, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sky->I, h_sky->I, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sky->Q, h_sky->Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sky->U, h_sky->U, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sky->V, h_sky->V, bytes, cudaMemcpyHostToDevice);

    // Copy number of sources to device.
    cudaMemcpy(&d_sky->num_sources, &h_sky->num_sources,
            sizeof(unsigned), cudaMemcpyHostToDevice);
}

void oskar_cuda_alloc_local_sky_d(int num_sources, oskar_SkyModelLocal_d* d_sky)
{
    size_t bytes = num_sources * sizeof(double);

    cudaMalloc((void**)&(d_sky->RA), bytes);
    cudaMalloc((void**)&(d_sky->Dec), bytes);
    cudaMalloc((void**)&(d_sky->I), bytes);
    cudaMalloc((void**)&(d_sky->Q), bytes);
    cudaMalloc((void**)&(d_sky->U), bytes);
    cudaMalloc((void**)&(d_sky->V), bytes);
    cudaMalloc((void**)&(d_sky->hor_l), bytes);
    cudaMalloc((void**)&(d_sky->hor_m), bytes);
    cudaMalloc((void**)&(d_sky->hor_n), bytes);
}


#ifdef __cplusplus
}
#endif
