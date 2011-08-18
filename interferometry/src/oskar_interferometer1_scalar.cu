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
#include "interferometry/cudak/oskar_cudak_correlator_scalar.h"
#include "interferometry/cudak/oskar_cudak_xyz_to_uvw.h"
#include "interferometry/oskar_compute_baselines.h"
#include "interferometry/oskar_xyz_to_uvw.h"


#include "math/cudak/oskar_cudak_dftw_3d_seq_out.h"
#include "math/cudak/oskar_cudak_mat_mul_cc.h"
#include "math/cudak/oskar_cudak_vec_mul_cr.h"

#include "sky/oskar_cuda_horizon_clip.h"
#include "sky/oskar_cuda_ra_dec_to_hor_lmn.h"
#include "sky/oskar_cuda_ra_dec_to_relative_lmn.h"

#include "station/oskar_evaluate_e_jones_2d_horizontal.h"

#include "sky/oskar_mjd_to_last_fast.h"


#include <cstdio>
#include <cmath>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SEC_PER_DAY 86400.0



#ifdef __cplusplus
extern "C" {
#endif

// Private functions.
//------------------------------------------------------------------------------
//void copy_telescope_to_gpu_d(const oskar_TelescopeModel* h_telescope,
//        oskar_TelescopeModel* hd_telescope);
//
//void copy_stations_to_gpu_d(const oskar_StationModel* h_stations,
//        const unsigned num_stations, oskar_StationModel* hd_stations);
//
//void copy_global_sky_to_gpu_d(const oskar_SkyModelGlobal_d* h_sky,
//        oskar_SkyModelGlobal_d* d_sky);
//
//void alloc_local_sky_d(int num_sources, oskar_SkyModelLocal_d* hd_sky);
//
//void alloc_beamforming_weights_work_buffer(const unsigned num_stations,
//        const oskar_StationModel* stations, double2** d_weights);
//
//void evaluate_e_jones(const unsigned num_stations,
//        const oskar_StationModel* hd_stations,
//        const oskar_SkyModelLocal_d* hd_sky, const double h_beam_l,
//        const double h_beam_m, double2* d_weights_work, double2* d_e_jones);
//
//void mult_e_jones_by_source_field_amp(const unsigned num_stations,
//        const oskar_SkyModelLocal_d* hd_sky, double2* d_e_jones);
//
//void evaluate_beam_horizontal_lm(double ra0_rad, double dec0_rad, double lst_rad,
//        double lat_rad, double* l, double* m);
//
//void correlate(const oskar_TelescopeModel* hd_telescope,
//        const oskar_SkyModelLocal_d* hd_sky, const double2* d_e_jones,
//        const double ra0_deg, const double dec0_deg, const double lst_start,
//        const int num_fringe_ave, const double dt_days, const double lambda,
//        const double bandwidtgh, double2* d_vis, double* work);
//
//------------------------------------------------------------------------------



int oskar_interferometer1_scalar_d(
        const oskar_TelescopeModel telescope, // NOTE: Already in ITRS coordinates
        const oskar_StationModel * stations,  // FIXME: PUT THESE IN WAVENUMBER UNITS (2 * pi * x / lambda)
        const oskar_SkyModelGlobal_d sky,
        const double ra0_rad,
        const double dec0_rad,
        const double obs_start_mjd_utc,
        const double obs_length_days,
        const unsigned num_vis_dumps,
        const unsigned num_vis_ave,
        const unsigned num_fringe_ave,
        const double freq,
        const double bandwidth,
        double2 * h_vis, // NOTE: Passed to the function as preallocated memory.
        double* h_u,     // NOTE: Passed to the function as preallocated memory.
        double* h_v,     // NOTE: Passed to the function as preallocated memory.
        double* h_w      // NOTE: Passed to the function as preallocated memory.
){
    // === Evaluate number of stations and number of baselines.
    const unsigned num_stations = telescope.num_antennas;
    const unsigned num_baselines = num_stations * (num_stations - 1) / 2;

    const double lambda = 299792458.0 / freq;

    // === Allocate device memory for telescope and transfer to device.
    oskar_TelescopeModel hd_telescope;
    copy_telescope_to_gpu_d(&telescope, &hd_telescope);

    // === Allocate memory to hold station uvw coordinates.
    double* station_u = (double*)malloc(num_stations * sizeof(double));
    double* station_v = (double*)malloc(num_stations * sizeof(double));
    double* station_w = (double*)malloc(num_stations * sizeof(double));

    // === Allocate device memory for antennas and transfer to the device.
    size_t mem_size = num_stations * sizeof(oskar_StationModel);
    oskar_StationModel * hd_stations = (oskar_StationModel*)malloc(mem_size);
    copy_stations_to_gpu_d(stations, num_stations, hd_stations);

    // === Allocate device memory for source model and transfer to device.
    oskar_SkyModelGlobal_d hd_sky_global;
    copy_global_sky_to_gpu_d(&sky, &hd_sky_global);

    // === Allocate local sky structure.
    oskar_SkyModelLocal_d hd_sky_local;
    alloc_local_sky_d(sky.num_sources, &hd_sky_local);

    // === Allocate device memory for station beam patterns.
    double2 * d_e_jones = NULL;
    size_t mem_e_jones = telescope.num_antennas * sky.num_sources * sizeof(double2);
    cudaMalloc((void**)&d_e_jones, mem_e_jones);

    // === Allocate device memory for beamforming weights buffer.
    double2* d_weights_work;
    alloc_beamforming_weights_work_buffer(telescope.num_antennas, stations,
            &d_weights_work);

    // === Allocate device memory for source positions in relative lmn coordinates.
    double *d_l, *d_m, *d_n;
    cudaMalloc((void**)&d_l, sky.num_sources * sizeof(double));
    cudaMalloc((void**)&d_m, sky.num_sources * sizeof(double));
    cudaMalloc((void**)&d_n, sky.num_sources * sizeof(double));

    // === Allocate device memory for visibilities.
    double2* d_vis;
    size_t mem_size_vis = num_baselines * sizeof(double2);
    cudaMalloc((void**)&d_vis, mem_size_vis);

    // === Allocate device memory for correlator work buffer.
    double* d_vis_work;
    size_t mem_size_work = (2 * sky.num_sources * num_stations + 3 * num_stations) * sizeof(double);
    cudaMalloc((void**)&d_vis_work, mem_size_work);

    // === Calculate time increments.
    unsigned total_samples   = num_vis_dumps * num_fringe_ave * num_vis_ave;
    double dt_days           = obs_length_days / total_samples;
    double dt_vis_days       = obs_length_days / num_vis_dumps;
    double dt_vis_ave_days   = dt_vis_days / num_vis_ave;
    double dt_vis_offset     = dt_vis_days / 2.0;
    double dt_vis_ave_offset = dt_vis_ave_days / 2.0;

    // === Loop over number of visibility snapshots.
    for (unsigned j = 0; j < num_vis_dumps; ++j)
    {
        // Start time for the visibility dump (in mjd utc)
        double t_vis_dump_start = obs_start_mjd_utc + (j * dt_vis_days);

        // Initialise visibilities for the dump to zero.
        cudaMemset((void*)d_vis, 0, mem_size_vis);

        // Loop over evaluations of the visibility average with changing
        // E-Jones within the dump.
        for (unsigned i = 0; i < num_vis_ave; ++i)
        {
            // Evaluate lst
            double t_ave_start = t_vis_dump_start + i * dt_vis_ave_days;
            double t_ave_mid   = t_ave_start + dt_vis_ave_offset;
            double lst = oskar_mjd_to_last_fast_d(t_ave_mid, telescope.longitude);

            // Find sources above horizon.
            oskar_cuda_horizon_clip_d(&hd_sky_global, lst, telescope.latitude,
                    &hd_sky_local);

            // Evaulate horizontal horizontal lm for the beam phase centre.
            double h_beam_l, h_beam_m;
            evaluate_beam_horizontal_lm(ra0_rad, dec0_rad, lst,
                    telescope.latitude, &h_beam_l, &h_beam_m);

            // Evaluate E-Jones for each source position per station
            evaluate_e_jones(num_stations, hd_stations, &hd_sky_local,
                    h_beam_l, h_beam_m, d_weights_work, d_e_jones);

            // Multiply e-jones by source brightness.
            mult_e_jones_by_source_field_amp(num_stations, &hd_sky_local, d_e_jones);

            // Convert source positions in ra, dec to lmn relative to the phase
            // centre.
            oskar_cuda_ra_dec_to_relative_lmn_d(hd_sky_local.num_sources,
                    hd_sky_local.RA, hd_sky_local.Dec, ra0_rad, dec0_rad,
                    d_l, d_m, d_n);

            // Correlator which updates phase matrix.
            double lst_start = oskar_mjd_to_last_fast_d(t_ave_start, telescope.longitude);
            oskar_cuda_correlator_scalar_d(num_stations, hd_telescope.antenna_x,
                    hd_telescope.antenna_y, hd_telescope.antenna_z,
                    hd_sky_local.num_sources, d_l, d_m, d_n,
                    (const double*)d_e_jones, ra0_rad, dec0_rad, lst_start,
                    num_fringe_ave, dt_days * SEC_PER_DAY, lambda * bandwidth,
                    (double*)d_vis, d_vis_work);
        }

        // copy back the vis dump into host memory.
        double2* vis = &h_vis[num_baselines * j];
        cudaMemcpy((void*)vis, (const void*)d_vis, mem_size_vis,
                cudaMemcpyDeviceToHost);

        // Evaluate baseline coordinates for the visibility dump.
        double* u = &h_u[num_baselines * j];
        double* v = &h_v[num_baselines * j];
        double* w = &h_w[num_baselines * j];
        double t_vis = t_vis_dump_start + dt_vis_offset;
        double lst_vis_dump = oskar_mjd_to_last_fast_d(t_vis, telescope.longitude);
        double ha_vis = lst_vis_dump - ra0_rad;
        oskar_xyz_to_uvw_d(num_stations, telescope.antenna_x, telescope.antenna_y,
                telescope.antenna_z, ha_vis, dec0_rad, station_u, station_v,
                station_w);
        oskar_compute_baselines_d(num_stations, station_u, station_v, station_w,
                u, v, w);
    }


    // free memory
    cudaFree(hd_telescope.antenna_x);
    cudaFree(hd_telescope.antenna_y);
    cudaFree(hd_telescope.antenna_z);
    free(station_u);
    free(station_v);
    free(station_w);
    for (unsigned i = 0; i < num_stations; ++i)
    {
        cudaFree(hd_stations[i].antenna_x);
        cudaFree(hd_stations[i].antenna_y);
    }
    free(hd_stations);

    cudaFree(hd_sky_global.RA);
    cudaFree(hd_sky_global.Dec);
    cudaFree(hd_sky_global.I);
    cudaFree(hd_sky_global.Q);
    cudaFree(hd_sky_global.U);
    cudaFree(hd_sky_global.V);

    cudaFree(hd_sky_local.RA);
    cudaFree(hd_sky_local.Dec);
    cudaFree(hd_sky_local.I);
    cudaFree(hd_sky_local.Q);
    cudaFree(hd_sky_local.U);
    cudaFree(hd_sky_local.V);
    cudaFree(hd_sky_local.hor_l);
    cudaFree(hd_sky_local.hor_m);
    cudaFree(hd_sky_local.hor_n);

    cudaFree(d_e_jones);
    cudaFree(d_weights_work);

    cudaFree(d_l);
    cudaFree(d_m);
    cudaFree(d_n);

    cudaFree(d_vis);

    cudaFree(d_vis_work);

    return (int)cudaPeekAtLastError();
}




void copy_telescope_to_gpu_d(const oskar_TelescopeModel * h_telescope,
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


void copy_stations_to_gpu_d(const oskar_StationModel * h_stations,
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


void copy_global_sky_to_gpu_d(const oskar_SkyModelGlobal_d* h_sky,
        oskar_SkyModelGlobal_d* hd_sky)
{
    // Allocate memory for arrays in structure.
    size_t bytes = h_sky->num_sources * sizeof(double);
    hd_sky->num_sources = h_sky->num_sources;

    cudaMalloc((void**)&(hd_sky->RA), bytes);
    cudaMalloc((void**)&(hd_sky->Dec), bytes);
    cudaMalloc((void**)&(hd_sky->I), bytes);
    cudaMalloc((void**)&(hd_sky->Q), bytes);
    cudaMalloc((void**)&(hd_sky->U), bytes);
    cudaMalloc((void**)&(hd_sky->V), bytes);

    // Copy arrays to device.
    cudaMemcpy(hd_sky->RA, h_sky->RA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->Dec, h_sky->Dec, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->I, h_sky->I, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->Q, h_sky->Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->U, h_sky->U, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hd_sky->V, h_sky->V, bytes, cudaMemcpyHostToDevice);
}

void alloc_local_sky_d(int num_sources, oskar_SkyModelLocal_d* d_sky)
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


void alloc_beamforming_weights_work_buffer(const unsigned num_stations,
        const oskar_StationModel* stations, double2** d_weights)
{
    unsigned num_antennas_max = 0;
    for (unsigned i = 0; i < num_stations; ++i)
    {
        if (stations[i].num_antennas > num_antennas_max)
            num_antennas_max = stations[i].num_antennas;
    }
    cudaMalloc((void**)d_weights, num_antennas_max * sizeof(double2));
}


void evaluate_e_jones(const unsigned num_stations,
        const oskar_StationModel * hd_stations,
        const oskar_SkyModelLocal_d * hd_sky, const double h_beam_l,
        const double h_beam_m, double2 * d_weights_work, double2 * d_e_jones)
{
    for (unsigned i = 0; i < num_stations; ++i)
    {
        double2 * d_e_jones_station = d_e_jones + i * hd_sky->num_sources;
        const oskar_StationModel * station = &hd_stations[i];
        oskar_evaluate_e_jones_2d_horizontal_d(station, h_beam_l, h_beam_m,
                hd_sky, d_weights_work, d_e_jones_station);
    }
}


void mult_e_jones_by_source_field_amp(const unsigned num_stations,
        const oskar_SkyModelLocal_d* hd_sky, double2 * d_e_jones)
{
    int num_sources = hd_sky->num_sources;
    int num_threads = 256;
    int num_blocks  = (int)ceil((double) num_sources / num_threads);

    for (unsigned i = 0; i < num_stations; ++i)
    {
        double2 * d_e_jones_station = d_e_jones + i * num_sources;
        oskar_cudak_vec_mul_cr_d <<< num_blocks, num_threads >>>
                (num_sources, d_e_jones_station, hd_sky->I, d_e_jones_station);
    }
}



void evaluate_beam_horizontal_lm(double ra0_rad, double dec0_rad, double lst_rad,
        double lat_rad, double * l, double * m)
{
    double ha = lst_rad - ra0_rad;
    *l = -cos(dec0_rad) * sin(ha);
    *m = cos(lat_rad) * sin(dec0_rad) - sin(lat_rad) * cos(dec0_rad) * cos(ha);
}


#ifdef __cplusplus
}
#endif
