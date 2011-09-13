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
#include "interferometry/oskar_compute_baselines.h"
#include "interferometry/oskar_xyz_to_uvw.h"

#include "sky/oskar_cuda_horizon_clip.h"
#include "sky/oskar_cuda_ra_dec_to_hor_lmn.h"
#include "sky/oskar_cuda_ra_dec_to_relative_lmn.h"
#include "sky/oskar_ra_dec_to_hor_lmn.h"

#include "station/oskar_mult_beampattern_by_sources.h"
#include "station/oskar_evaluate_station_beam.h"

#include "sky/oskar_mjd_to_last_fast.h"

#include <cstdio>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
void alloc_beamforming_weights_buffer_d(const unsigned num_stations,
        const oskar_StationModel_d* stations, double2** d_weights);

void alloc_beamforming_weights_buffer_f(const unsigned num_stations,
        const oskar_StationModel_f* stations, float2** d_weights);
// =============================================================================


int oskar_interferometer1_scalar_d(
        const oskar_TelescopeModel_d telescope,
        const oskar_StationModel_d * stations,
        const oskar_SkyModelGlobal_d sky,
        const double ra0_rad,
        const double dec0_rad,
        const double obs_start_mjd_utc,
        const double obs_length_days,
        const unsigned num_vis_dumps,
        const unsigned num_vis_ave,
        const unsigned num_fringe_ave,
        const double frequency,
        const double bandwidth,
        const bool disable_e_jones,
        oskar_VisData_d* h_vis
){
    const double sec_per_day = 86400.0;

    // === Evaluate number of stations and number of baselines.
    const unsigned num_stations = telescope.num_antennas;
    const unsigned num_baselines = num_stations * (num_stations - 1) / 2;

    const double lambda = 299792458.0 / frequency;
    const double wavenumber = 2.0 * M_PI / lambda;

    // === Allocate device memory for telescope and transfer to device.
    oskar_TelescopeModel_d hd_telescope;
    oskar_copy_telescope_to_device_d(&telescope, &hd_telescope);

    // === Scale telescope coordinates to wavenumber units.
    oskar_scale_device_telescope_coords_d(&hd_telescope, wavenumber);

    // === Allocate memory to hold station uvw coordinates.
    double* station_u = (double*)malloc(num_stations * sizeof(double));
    double* station_v = (double*)malloc(num_stations * sizeof(double));
    double* station_w = (double*)malloc(num_stations * sizeof(double));

    // === Allocate device memory for antennas and transfer to the device.
    size_t mem_size = num_stations * sizeof(oskar_StationModel_d);
    oskar_StationModel_d* hd_stations = (oskar_StationModel_d*)malloc(mem_size);
    oskar_copy_stations_to_device_d(stations, num_stations, hd_stations);

    // === Scale station coordinates to wavenumber units.
    oskar_scale_station_coords_d(num_stations, hd_stations, wavenumber);

    // === Allocate device memory for source model and transfer to device.
    oskar_SkyModelGlobal_d hd_sky_global;
    for (int i = 0; i < sky.num_sources; ++i)
        sky.I[i] = sqrt(sky.I[i]);
    oskar_copy_gobal_sky_to_device_d(&sky, &hd_sky_global);

    // === Allocate local sky structure.
    oskar_SkyModelLocal_d hd_sky_local;
    oskar_allocate_device_local_sky_d(sky.num_sources, &hd_sky_local);

    // === Allocate device memory for station beam patterns.
    double2 * d_e_jones = NULL;
    size_t mem_e_jones = telescope.num_antennas * sky.num_sources * sizeof(double2);
    cudaMalloc((void**)&d_e_jones, mem_e_jones);

    // === Allocate device memory for beamforming weights buffer.
    double2* d_weights_work;
    alloc_beamforming_weights_buffer_d(telescope.num_antennas, stations,
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
    double* d_work_uvw;
    cudaMalloc((void**)&d_work_uvw, 3 * num_stations * sizeof(double));
    double2* d_work_k;
    cudaMalloc((void**)&d_work_k, num_stations * hd_sky_global.num_sources *  sizeof(double2));

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
        printf("--> simulating visibility snapshot %i of %i ...\n", j+1,
                num_vis_dumps);

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

            if (hd_sky_local.num_sources == 0)
                fprintf(stderr, "WARNING: no sources above horizon! (this will fail!)\n");

            // Evaluate horizontal lm for the beam phase centre.
            double h_beam_l, h_beam_m, h_beam_n;
            oskar_ra_dec_to_hor_lmn_d(1, &ra0_rad, &dec0_rad, lst,
                    telescope.latitude, &h_beam_l, &h_beam_m, &h_beam_n);

            // Evaluate E-Jones for each source position per station
            oskar_evaluate_station_beams_d(num_stations, hd_stations,
                    &hd_sky_local, h_beam_l, h_beam_m, d_weights_work,
                    disable_e_jones, telescope.identical_stations, d_e_jones);

            // Multiply e-jones by source brightness.
            oskar_mult_beampattern_by_source_field_amp_d(num_stations,
                    &hd_sky_local, d_e_jones);

            // Convert source positions in ra, dec to lmn relative to the phase
            // centre.
            oskar_cuda_ra_dec_to_relative_lmn_d(hd_sky_local.num_sources,
                    hd_sky_local.RA, hd_sky_local.Dec, ra0_rad, dec0_rad,
                    d_l, d_m, d_n);

            // Correlator which updates phase matrix.
            double lst_start = oskar_mjd_to_last_fast_d(t_ave_start,
                    telescope.longitude);
            oskar_cuda_correlator_scalar_d(num_stations, hd_telescope.antenna_x,
                    hd_telescope.antenna_y, hd_telescope.antenna_z,
                    hd_sky_local.num_sources, d_l, d_m, d_n,
                    d_e_jones, ra0_rad, dec0_rad, lst_start,
                    num_fringe_ave, dt_days * sec_per_day, lambda * bandwidth,
                    d_work_k, d_work_uvw, d_vis);
        }

        // copy back the vis dump into host memory.
        cudaMemcpy(&(h_vis->amp[num_baselines * j]), d_vis, mem_size_vis,
                cudaMemcpyDeviceToHost);

        // Evaluate baseline coordinates for the visibility dump.
        double* u = &h_vis->u[num_baselines * j];
        double* v = &h_vis->v[num_baselines * j];
        double* w = &h_vis->w[num_baselines * j];
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
    oskar_free_device_telescope_d(&hd_telescope);
    oskar_free_device_global_sky_d(&hd_sky_global);
    oskar_free_device_local_sky_d(&hd_sky_local);
    free(station_u);
    free(station_v);
    free(station_w);
    for (unsigned i = 0; i < num_stations; ++i)
    {
        cudaFree(hd_stations[i].antenna_x);
        cudaFree(hd_stations[i].antenna_y);
    }
    free(hd_stations);
    cudaFree(d_e_jones);
    cudaFree(d_weights_work);
    cudaFree(d_l);
    cudaFree(d_m);
    cudaFree(d_n);
    cudaFree(d_vis);
    cudaFree(d_work_uvw);
    cudaFree(d_work_k);

    // Catch any CUDA errors and return
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "CUDA ERROR [%i] from oskar_interferometer1_scalar(): %s\n",
                (int)error, cudaGetErrorString(error));
    }
    return (int)error;
}



int oskar_interferometer1_scalar_f(
        const oskar_TelescopeModel_f telescope,
        const oskar_StationModel_f * stations,
        const oskar_SkyModelGlobal_f sky,
        const float ra0_rad,
        const float dec0_rad,
        const float obs_start_mjd_utc,
        const float obs_length_days,
        const unsigned num_vis_dumps,
        const unsigned num_vis_ave,
        const unsigned num_fringe_ave,
        const float frequency,
        const float bandwidth,
        const bool disable_e_jones,
        oskar_VisData_f* h_vis
)
{
    const float sec_per_day = 86400.0f;

    // === Evaluate number of stations and number of baselines.
    const unsigned num_stations = telescope.num_antennas;
    const unsigned num_baselines = num_stations * (num_stations - 1) / 2;

    const float lambda     = 299792458.0f / frequency;
    const float wavenumber = 2.0f * M_PI / lambda;

    // === Allocate device memory for telescope and transfer to device.
    oskar_TelescopeModel_f hd_telescope;
    oskar_copy_telescope_to_device_f(&telescope, &hd_telescope);

    // === Scale telescope coordinates to wavenumber units.
    oskar_scale_device_telescope_coords_f(&hd_telescope, (float)wavenumber);

    // === Allocate memory to hold station uvw coordinates.
    float* station_u = (float*)malloc(num_stations * sizeof(float));
    float* station_v = (float*)malloc(num_stations * sizeof(float));
    float* station_w = (float*)malloc(num_stations * sizeof(float));

    // === Allocate device memory for antennas and transfer to the device.
    size_t mem_size = num_stations * sizeof(oskar_StationModel_f);
    oskar_StationModel_f* hd_stations = (oskar_StationModel_f*)malloc(mem_size);
    oskar_copy_stations_to_device_f(stations, num_stations, hd_stations);

    // === Scale station coordinates to wavenumber units.
    oskar_scale_station_coords_f(num_stations, hd_stations, (float)wavenumber);

    // === Allocate device memory for source model and transfer to device.
    oskar_SkyModelGlobal_f hd_sky_global;
    for (int i = 0; i < sky.num_sources; ++i)
        sky.I[i] = sqrt(sky.I[i]);
    oskar_copy_gobal_sky_to_device_f(&sky, &hd_sky_global);

    // === Allocate local sky structure.
    oskar_SkyModelLocal_f hd_sky_local;
    oskar_allocate_device_local_sky_f(sky.num_sources, &hd_sky_local);

    // === Allocate device memory for station beam patterns.
    float2 * d_e_jones = NULL;
    size_t mem_e_jones = telescope.num_antennas * sky.num_sources * sizeof(float2);
    cudaMalloc((void**)&d_e_jones, mem_e_jones);

    // === Allocate device memory for beamforming weights buffer.
    float2* d_weights_work;
    alloc_beamforming_weights_buffer_f(telescope.num_antennas, stations,
            &d_weights_work);

    // === Allocate device memory for source positions in relative lmn coordinates.
    float *d_l, *d_m, *d_n;
    cudaMalloc((void**)&d_l, sky.num_sources * sizeof(float));
    cudaMalloc((void**)&d_m, sky.num_sources * sizeof(float));
    cudaMalloc((void**)&d_n, sky.num_sources * sizeof(float));

    // === Allocate device memory for visibilities.
    float2* d_vis;
    size_t mem_size_vis = num_baselines * sizeof(float2);
    cudaMalloc((void**)&d_vis, mem_size_vis);

    // === Allocate device memory for correlator work buffer.
    float* d_work_uvw;
    cudaMalloc((void**)&d_work_uvw, 3 * num_stations * sizeof(float));
    float2* d_work_k;
    cudaMalloc((void**)&d_work_k, num_stations * hd_sky_global.num_sources *  sizeof(float2));

    // === Calculate time increments.
    unsigned total_samples  = num_vis_dumps * num_fringe_ave * num_vis_ave;
    float dt_days           = obs_length_days / total_samples;
    float dt_vis_days       = obs_length_days / num_vis_dumps;
    float dt_vis_ave_days   = dt_vis_days / num_vis_ave;
    float dt_vis_offset     = dt_vis_days / 2.0;
    float dt_vis_ave_offset = dt_vis_ave_days / 2.0;

    // === Loop over number of visibility snapshots.
    for (unsigned j = 0; j < num_vis_dumps; ++j)
    {
        printf("--> simulating visibility snapshot %i of %i ...\n", j+1, num_vis_dumps);

        // Start time for the visibility dump (in mjd utc)
        float t_vis_dump_start = obs_start_mjd_utc + (j * dt_vis_days);

        // Initialise visibilities for the dump to zero.
        cudaMemset((void*)d_vis, 0, mem_size_vis);

        // Loop over evaluations of the visibility average with changing
        // E-Jones within the dump.
        for (unsigned i = 0; i < num_vis_ave; ++i)
        {
            // Evaluate lst
            float t_ave_start = t_vis_dump_start + i * dt_vis_ave_days;
            float t_ave_mid   = t_ave_start + dt_vis_ave_offset;
            float lst = oskar_mjd_to_last_fast_d(t_ave_mid, telescope.longitude);

            // Find sources above horizon.
            oskar_cuda_horizon_clip_f(&hd_sky_global, lst, telescope.latitude,
                    &hd_sky_local);

            if (hd_sky_local.num_sources == 0)
            {
                fprintf(stderr, "WARNING: no sources above horizon! (this will fail!)\n");

            }

            // Evaluate horizontal lm for the beam phase centre.
            float h_beam_l, h_beam_m, h_beam_n;
            oskar_ra_dec_to_hor_lmn_f(1, &ra0_rad, &dec0_rad, lst,
                    telescope.latitude, &h_beam_l, &h_beam_m, &h_beam_n);

            // Evaluate E-Jones for each source position per station
            oskar_evaluate_station_beams_f(num_stations, hd_stations,
                    &hd_sky_local, h_beam_l, h_beam_m, d_weights_work,
                    disable_e_jones, telescope.identical_stations, d_e_jones);

            // Multiply e-jones by source brightness.
            oskar_mult_beampattern_by_source_field_amp_f(num_stations,
                    &hd_sky_local, d_e_jones);

            // Convert source positions in ra, dec to lmn relative to the phase
            // centre.
            oskar_cuda_ra_dec_to_relative_lmn_f(hd_sky_local.num_sources,
                    hd_sky_local.RA, hd_sky_local.Dec, ra0_rad, dec0_rad,
                    d_l, d_m, d_n);

            // Correlator which updates phase matrix.
            float lst_start = oskar_mjd_to_last_fast_f(t_ave_start,
                    telescope.longitude);
            oskar_cuda_correlator_scalar_f((int)num_stations,
                    hd_telescope.antenna_x, hd_telescope.antenna_y,
                    hd_telescope.antenna_z, hd_sky_local.num_sources,
                    d_l, d_m, d_n, d_e_jones, ra0_rad, dec0_rad, lst_start,
                    num_fringe_ave, dt_days * sec_per_day, lambda * bandwidth,
                    d_work_k, d_work_uvw, d_vis);
        }

        // copy back the vis dump into host memory.
        cudaMemcpy(&(h_vis->amp[num_baselines * j]), d_vis, mem_size_vis,
                cudaMemcpyDeviceToHost);

        // Evaluate baseline coordinates for the visibility dump.
        float* u = &(h_vis->u[num_baselines * j]);
        float* v = &(h_vis->v[num_baselines * j]);
        float* w = &(h_vis->w[num_baselines * j]);
        float t_vis = t_vis_dump_start + dt_vis_offset;
        float lst_vis_dump = oskar_mjd_to_last_fast_d(t_vis, telescope.longitude);
        float ha_vis = lst_vis_dump - ra0_rad;
        oskar_xyz_to_uvw_f(num_stations, telescope.antenna_x, telescope.antenna_y,
                telescope.antenna_z, ha_vis, dec0_rad, station_u, station_v,
                station_w);
        oskar_compute_baselines_f(num_stations, station_u, station_v, station_w,
                u, v, w);
    }


    // free memory
    oskar_free_device_telescope_f(&hd_telescope);
    free(station_u);
    free(station_v);
    free(station_w);
    for (unsigned i = 0; i < num_stations; ++i)
    {
        cudaFree(hd_stations[i].antenna_x);
        cudaFree(hd_stations[i].antenna_y);
    }
    free(hd_stations);

    oskar_free_device_global_sky_f(&hd_sky_global);
    oskar_free_device_local_sky_f(&hd_sky_local);

    cudaFree(d_e_jones);
    cudaFree(d_weights_work);

    cudaFree(d_l);
    cudaFree(d_m);
    cudaFree(d_n);

    cudaFree(d_vis);

    cudaFree(d_work_uvw);
    cudaFree(d_work_k);

    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "CUDA ERROR [%i] from oskar_interferometer1_scalar(): %s\n",
                (int)error, cudaGetErrorString(error));
    }

    return (int)error;
}






//==============================================================================
void alloc_beamforming_weights_buffer_d(const unsigned num_stations,
        const oskar_StationModel_d* stations, double2** d_weights)
{
    unsigned num_antennas_max = 0;
    for (unsigned i = 0; i < num_stations; ++i)
    {
        if (stations[i].num_antennas > num_antennas_max)
            num_antennas_max = stations[i].num_antennas;
    }
    cudaMalloc((void**)d_weights, num_antennas_max * sizeof(double2));
}



void alloc_beamforming_weights_buffer_f(const unsigned num_stations,
        const oskar_StationModel_f* stations, float2** d_weights)
{
    unsigned num_antennas_max = 0;
    for (unsigned i = 0; i < num_stations; ++i)
    {
        if (stations[i].num_antennas > num_antennas_max)
            num_antennas_max = stations[i].num_antennas;
    }
    cudaMalloc((void**)d_weights, num_antennas_max * sizeof(float2));
}
//==============================================================================


#ifdef __cplusplus
}
#endif
