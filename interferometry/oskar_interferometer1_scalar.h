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

#ifndef OSKAR_CUDA_INTERFEROMETER1_SCALAR_H_
#define OSKAR_CUDA_INTERFEROMETER1_SCALAR_H_

/**
 * @file oskar_cuda_interferometer1_scalar.h
 */

#include "oskar_windows.h"

#include "sky/oskar_SkyModel.h"
#include "station/oskar_StationModel.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "utility/oskar_vector_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 *
 * @details
 *
 */
DllExport
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
        double2 * h_vis, // NOTE: Passed to the function as preallocated memory.
        double* h_u,     // NOTE: Passed to the function as preallocated memory.
        double* h_v,     // NOTE: Passed to the function as preallocated memory.
        double* h_w      // NOTE: Passed to the function as preallocated memory.
);


void copy_telescope_to_gpu_d(const oskar_TelescopeModel* h_telescope,
        oskar_TelescopeModel* hd_telescope);

void copy_stations_to_gpu_d(const oskar_StationModel* h_stations,
        const unsigned num_stations, oskar_StationModel* hd_stations);

void copy_global_sky_to_gpu_d(const oskar_SkyModelGlobal_d* h_sky,
        oskar_SkyModelGlobal_d* d_sky);

void alloc_local_sky_d(int num_sources, oskar_SkyModelLocal_d* hd_sky);

void alloc_beamforming_weights_work_buffer(const unsigned num_stations,
        const oskar_StationModel* stations, double2** d_weights);

void evaluate_e_jones(const unsigned num_stations,
        const oskar_StationModel* hd_stations,
        const oskar_SkyModelLocal_d* hd_sky, const double h_beam_l,
        const double h_beam_m, double2* d_weights_work, double2* d_e_jones);

void mult_e_jones_by_source_field_amp(const unsigned num_stations,
        const oskar_SkyModelLocal_d* hd_sky, double2* d_e_jones);

void evaluate_beam_horizontal_lm(double ra0_rad, double dec0_rad, double lst_rad,
        double lat_rad, double* l, double* m);

void correlate(const oskar_TelescopeModel* hd_telescope,
        const oskar_SkyModelLocal_d* hd_sky, const double2* d_e_jones,
        const double ra0_deg, const double dec0_deg, const double lst_start,
        const int num_fringe_ave, const double dt_days, const double lambda,
        const double bandwidtgh, double2* d_vis, double* work);


#ifdef __cplusplus
}
#endif

#endif // OSKAR_CUDA_INTERFEROMETER1_SCALAR_H_
