/*
 * Copyright (c) 2012, The University of Oxford
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

#include <cuda_runtime_api.h>

#include "interferometry/oskar_interferometer.h"
#include "interferometry/oskar_correlate.h"
#include "interferometry/oskar_evaluate_jones_K.h"
#include "interferometry/oskar_evaluate_uvw_station.h"
#include "math/oskar_Jones.h"
#include "math/oskar_jones_join.h"
#include "math/oskar_jones_set_size.h"
#include "sky/oskar_evaluate_jones_R.h"
#include "sky/oskar_mjd_to_gast_fast.h"
#include "sky/oskar_sky_model_horizon_clip.h"
#include "station/oskar_evaluate_jones_E.h"
#include "utility/oskar_Device_curand_state.h"
#include "utility/oskar_log_message.h"
#include "utility/oskar_log_warning.h"
#include "utility/oskar_mem_clear_contents.h"
#include "utility/oskar_mem_insert.h"
#include "utility/oskar_mem_scale_real.h"
#include <cstdio>

extern "C"
int oskar_interferometer(oskar_Mem* vis_amp, oskar_Log* log,
        const oskar_SkyModel* sky, const oskar_TelescopeModel* telescope,
        const oskar_Settings* settings, double frequency, int chunk_index,
        int num_sky_chunks)
{
    int status = OSKAR_SUCCESS;
    int device_id = 0;
    size_t mem_free = 0, mem_total = 0;
    cudaDeviceProp device_prop;

    // Always clear the output array to ensure that all visibilities are zero
    // if there are never any visible sources in the sky model.
    oskar_mem_clear_contents(vis_amp, &status);
    if (status) return status;

    // Get the current device ID.
    cudaGetDevice(&device_id);

    // Check if sky model is empty.
    if (sky->num_sources == 0)
    {
        oskar_log_warning(log, "No sources in sky model. Skipping "
                "Measurement Equation evaluation.");
        return OSKAR_SUCCESS;
    }

    // Copy telescope model and sky model for frequency scaling.
    oskar_TelescopeModel tel_gpu(telescope, OSKAR_LOCATION_GPU);
    oskar_SkyModel sky_gpu(sky, OSKAR_LOCATION_GPU);

    // Scale GPU telescope coordinates by wavenumber.
    status = tel_gpu.multiply_by_wavenumber(frequency);
    if (status) return status;

    // Scale by spectral index.
    status = sky_gpu.scale_by_spectral_index(frequency);
    if (status) return status;

    // Initialise blocks of Jones matrices and visibilities.
    int type = sky_gpu.type();
    int n_stations = tel_gpu.num_stations;
    int n_baselines = n_stations * (n_stations - 1) / 2;
    int n_sources = sky_gpu.num_sources;
    int complex_scalar = type | OSKAR_COMPLEX;
    int complex_matrix = type | OSKAR_COMPLEX | OSKAR_MATRIX;
    oskar_Jones J(complex_matrix, OSKAR_LOCATION_GPU, n_stations, n_sources);
    oskar_Jones R(complex_matrix, OSKAR_LOCATION_GPU, n_stations, n_sources);
    oskar_Jones E(complex_matrix, OSKAR_LOCATION_GPU, n_stations, n_sources);
    oskar_Jones K(complex_scalar, OSKAR_LOCATION_GPU, n_stations, n_sources);
    oskar_Mem vis(complex_matrix, OSKAR_LOCATION_GPU, n_baselines);
    oskar_Mem u(type, OSKAR_LOCATION_GPU, n_stations, true);
    oskar_Mem v(type, OSKAR_LOCATION_GPU, n_stations, true);
    oskar_Mem w(type, OSKAR_LOCATION_GPU, n_stations, true);
    oskar_WorkStationBeam work(type, OSKAR_LOCATION_GPU);

    // Declare a local sky model of sufficient size for the horizon clip.
    oskar_SkyModel local_sky(type, OSKAR_LOCATION_GPU, n_sources);

    // Initialise the random number generator.
    // Note: This is reset to the same sequence per sky chunk and per channel.
    // This is required so that when splitting the sky into chunks or channels
    // antennas still have the same error value for the given time and seed.
    oskar_Device_curand_state curand_state(telescope->max_station_size);
    curand_state.init(telescope->seed_time_variable_station_element_errors);

    // Get time increments.
    int num_vis_dumps        = settings->obs.num_time_steps;
    double obs_start_mjd_utc = settings->obs.start_mjd_utc;
    int num_vis_ave          = settings->interferometer.num_vis_ave;
    int num_fringe_ave       = settings->interferometer.num_fringe_ave;
    double dt_dump   = settings->obs.dt_dump_days;
    double dt_ave    = dt_dump / settings->interferometer.num_vis_ave;
    double dt_fringe = dt_ave / settings->interferometer.num_fringe_ave;

    // Start simulation.
    for (int j = 0; j < num_vis_dumps; ++j)
    {
        // Start time for the visibility dump, in MJD(UTC).
        double t_dump = obs_start_mjd_utc + j * dt_dump;
        double gast = oskar_mjd_to_gast_fast(t_dump + dt_dump / 2.0);

        // Initialise visibilities for the dump to zero.
        oskar_mem_clear_contents(&vis, &status);

        // Compact sky model to temporary.
        oskar_sky_model_horizon_clip(&local_sky, &sky_gpu, &tel_gpu, gast,
                &work, &status);

        // Record number of visible sources in this snapshot.
        oskar_log_message(log, 1, "Snapshot %4d/%d, chunk %4d/%d, "
                "device %d [%d sources]", j+1, num_vis_dumps, chunk_index+1,
                num_sky_chunks, device_id, local_sky.num_sources);

        // Skip iteration if no sources above horizon.
        if (local_sky.num_sources == 0) continue;

        // Set dimensions of Jones matrices (this is not a resize!).
        oskar_jones_set_size(&J, n_stations, local_sky.num_sources, &status);
        oskar_jones_set_size(&R, n_stations, local_sky.num_sources, &status);
        oskar_jones_set_size(&E, n_stations, local_sky.num_sources, &status);
        oskar_jones_set_size(&K, n_stations, local_sky.num_sources, &status);
        if (status) return status;

        // Average snapshot.
        for (int i = 0; i < num_vis_ave; ++i)
        {
            // Evaluate Greenwich Apparent Sidereal Time.
            double t_ave = t_dump + i * dt_ave;
            double gast = oskar_mjd_to_gast_fast(t_ave + dt_ave / 2);

            // Evaluate parallactic angle rotation (Jones R).
            status = oskar_evaluate_jones_R(&R, &local_sky, &tel_gpu, gast);
            if (status) return status;

            // Evaluate station beam (Jones E).
            status = oskar_evaluate_jones_E(&E, &local_sky, &tel_gpu, gast,
                    &work, &curand_state);
            if (status) return status;

            // Join Jones matrices (R = E * R).
            oskar_jones_join(&R, &E, &R, &status);
            if (status) return status;

            for (int k = 0; k < num_fringe_ave; ++k)
            {
                // Evaluate Greenwich Apparent Sidereal Time.
                double t_fringe = t_ave + k * dt_fringe;
                double gast = oskar_mjd_to_gast_fast(t_fringe + dt_fringe / 2);

                // Evaluate station u,v,w coordinates.
                oskar_evaluate_uvw_station(&u, &v, &w, tel_gpu.num_stations,
                        &tel_gpu.station_x, &tel_gpu.station_y,
                        &tel_gpu.station_z, tel_gpu.ra0_rad, tel_gpu.dec0_rad,
                        gast, &status);
                if (status) return status;

                // Evaluate interferometer phase (Jones K).
                status = oskar_evaluate_jones_K(&K, &local_sky, &u, &v, &w);
                if (status) return status;

                // Join Jones matrices (J = K * R).
                oskar_jones_join(&J, &K, &R, &status);
                if (status) return status;

                // Form baseline pairs.
                status = oskar_correlate(&vis, &J, &tel_gpu, &local_sky, &u, &v);
                if (status) return status;
            }
        }

        // Divide visibilities by number of averages.
        oskar_mem_scale_real(&vis, 1.0 / (num_fringe_ave * num_vis_ave),
                &status);

        // Add visibilities to global data.
        oskar_mem_insert(vis_amp, &vis, j * n_baselines, &status);
        if (status) return status;
    }

    // Record GPU memory usage.
    cudaMemGetInfo(&mem_free, &mem_total);
    cudaGetDeviceProperties(&device_prop, device_id);
    oskar_log_message(log, 1, "Memory on device %d [%s] is %.1f%% used.",
            device_id, device_prop.name,
            100.0 * (1.0 - ((double)mem_free / (double)mem_total)));

    return OSKAR_SUCCESS;
}
