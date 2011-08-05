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

#include "interferometry/oskar_cuda_interferometer1_scalar.h"
#include "interferometry/oskar_cuda_correlator_scalar.h"

#include "math/cudak/oskar_math_cudak_dftw_3d_seq_out.h"
#include "math/cudak/oskar_math_cudak_mat_mul_cc.h"

#include "interferometry/cudak/oskar_cudak_correlator.h"
#include "interferometry/cudak/oskar_cudak_xyz2uvw.h"

#include <cstdio>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SEC_PER_DAY 86400.0

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
void oskar_cudad_copy_telescope_to_gpu(const struct TelescopeModel * h_telescope,
        struct TelescopeModel * d_telescope);

void oskar_cudad_copy_stations_to_gpu(const struct StationModel * h_stations,
        const unsigned num_stations, struct StationModel * d_stations);

void oskar_cudad_copy_sky_to_gpu(const struct SkyModel * h_sky,
        struct SkyModel * d_sky);
//------------------------------------------------------------------------------



int oskar_cudad_interferometer1_scalar(

        const struct TelescopeModel telescope, // NOTE: In ITRS coordinates

        const struct StationModel * stations,

        const struct SkyModel sky,

        const double ra0_rads,
        const double dec0_rads,

        const double start_date_utc,
        const unsigned nsdt,
        const double sdt,

        const double lambda_bandwidth,

        double * vis // FIXME float2?
){
    cudaError_t cuda_error = cudaSuccess;

    // === Evaluate number of stations and number of baselines.
    const unsigned num_stations = telescope.num_antennas;
    const unsigned num_baselines = num_stations * (num_stations - 1) / 2;

    // === Allocate device memory for telescope and transfer to device.
    struct TelescopeModel d_telescope;
    oskar_cudad_copy_telescope_to_gpu(&telescope, &d_telescope);

    // === Allocate device memory for antennas and transfer to the device.
    size_t mem_size = num_stations * sizeof(StationModel);
    struct StationModel * d_stations = (StationModel*)malloc(mem_size);
    oskar_cudad_copy_stations_to_gpu(stations, num_stations, d_stations);

    // === Allocate device memory for source model and transfer to device.
    struct SkyModel d_sky;
    oskar_cudad_copy_sky_to_gpu(&sky, &d_sky);









    // TODO: Transform the station positions to the local equatorial system.
    //[X, Y, Z] = horizon_plane_to_itrs(Xh, Yh, lat);

    int num_vis_snapshots = 0;
    int num_vis_averages = 0;


    // 4. Loop over number of visibility snapshots.
    for (int j = 0; j < num_vis_snapshots; ++j)
    {
        // 5. Evaluate LST from UTC.
        // TODO

        // 6. Loop over evaluations of the visibility average with changing E-Jones
        // within the dump
        for (int i = 0; i < num_vis_averages; ++i)
        {
            // 6. Find sources above horizon.
            // TODO

            // 7. Evaluate E-Jones for each source position per station.
            // TODO: optimisation if all stations are the same?
            //for each station
            //      oskar_cudad_bp2hc() <=== move this to beamforming folder.

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




void oskar_cudad_copy_telescope_to_gpu(const struct TelescopeModel * h_telescope,
        struct TelescopeModel * d_telescope)
{
    size_t mem_size = h_telescope->num_antennas * sizeof(double);

    d_telescope->num_antennas = h_telescope->num_antennas;

    cudaMalloc((void**)&(d_telescope->antenna_x), mem_size);
    cudaMalloc((void**)&(d_telescope->antenna_y), mem_size);
    cudaMalloc((void**)&(d_telescope->antenna_z), mem_size);

    cudaMemcpy(d_telescope->antenna_x, h_telescope->antenna_x, mem_size,
            cudaMemcpyHostToDevice);
    cudaMemcpy(d_telescope->antenna_y, h_telescope->antenna_y, mem_size,
            cudaMemcpyHostToDevice);
    cudaMemcpy(d_telescope->antenna_z, h_telescope->antenna_z, mem_size,
            cudaMemcpyHostToDevice);
}


void oskar_cudad_copy_stations_to_gpu(const struct StationModel * h_stations,
        const unsigned num_stations, struct StationModel * d_stations)
{
    // Allocate and copy memory for each station.
    for (unsigned i = 0; i < num_stations; ++i)
    {
        d_stations[i].num_antennas = h_stations[i].num_antennas;

        size_t mem_size = d_stations[i].num_antennas * sizeof(double);
        cudaMalloc((void**)&(d_stations[i].antenna_x), mem_size);
        cudaMalloc((void**)&(d_stations[i].antenna_y), mem_size);

        cudaMemcpy(d_stations[i].antenna_x, h_stations[i].antenna_x, mem_size,
                cudaMemcpyHostToDevice);
        cudaMemcpy(d_stations[i].antenna_y, h_stations[i].antenna_y, mem_size,
                cudaMemcpyHostToDevice);
    }
}


void oskar_cudad_copy_sky_to_gpu(const struct SkyModel * h_sky,
        struct SkyModel * d_sky)
{
    // TODO: work out what needs to be in here...
}


#ifdef __cplusplus
}
#endif
