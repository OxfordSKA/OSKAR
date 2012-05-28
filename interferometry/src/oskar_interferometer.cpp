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
#include "interferometry/oskar_evaluate_station_uvw.h"
#include "math/oskar_Jones.h"
#include "math/oskar_jones_join.h"
#include "sky/oskar_evaluate_jones_R.h"
#include "sky/oskar_mjd_to_gast_fast.h"
#include "sky/oskar_sky_model_horizon_clip.h"
#include "station/oskar_evaluate_jones_E.h"
#include "utility/oskar_Device_curand_state.h"
#include "utility/oskar_log_message.h"
#include "utility/oskar_log_warning.h"
#include <cstdio>

extern "C"
int oskar_interferometer(oskar_Mem* vis_amp, oskar_Log* log,
        const oskar_SkyModel* sky, const oskar_TelescopeModel* telescope,
        const oskar_SettingsTime* times, double frequency, int chunk_index,
        int num_sky_chunks)
{
    int err = 0;
    int device_id = 0;

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
    err = tel_gpu.multiply_by_wavenumber(frequency);
    if (err) return err;

    // Scale by spectral index.
    err = sky_gpu.scale_by_spectral_index(frequency);
    if (err) return err;

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
    oskar_Work work(type, OSKAR_LOCATION_GPU);

    // Declare a local sky model of sufficient size for the horizon clip.
    oskar_SkyModel local_sky(type, OSKAR_LOCATION_GPU, n_sources);

    // Initialise the random number generator.
    // Note: This is reset to the same sequence per sky chunk and per channel.
    // This is required so that when splitting the sky into chunks or channels
    // antennas still have the same error value for the given time and seed.
    oskar_Device_curand_state curand_state(telescope->max_station_size);
    curand_state.init(telescope->seed_time_variable_errors);

    // Get time increments.
    int num_vis_dumps        = times->num_time_steps;
    int num_vis_ave          = times->num_vis_ave;
    int num_fringe_ave       = times->num_fringe_ave;
    double obs_start_mjd_utc = times->obs_start_mjd_utc;
    double dt_dump           = times->dt_dump_days;
    double dt_ave            = times->dt_ave_days;
    double dt_fringe         = times->dt_fringe_days;

    // Record GPU memory usage.
    size_t mem_free, mem_total;
    cudaDeviceProp device_prop;
    cudaMemGetInfo(&mem_free, &mem_total);
    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&device_prop, device_id);
    oskar_log_message(log, 1, "Memory on device %d [%s] is %.1f%% used.",
            device_id, device_prop.name,
            100 * (1.0 - ((float)mem_free / (float)mem_total)));

    // Start simulation.
    for (int j = 0; j < num_vis_dumps; ++j)
    {
        // Start time for the visibility dump, in MJD(UTC).
        double t_dump = obs_start_mjd_utc + j * dt_dump;
        double gast = oskar_mjd_to_gast_fast(t_dump + dt_dump / 2.0);

        // Initialise visibilities for the dump to zero.
        err = vis.clear_contents();
        if (err) return err;

        // Compact sky model to temporary.
        err = oskar_sky_model_horizon_clip(&local_sky, &sky_gpu, &tel_gpu,
                gast, &work);

        // Record number of visible sources in this snapshot.
        oskar_log_message(log, 1, "Snapshot %4d/%d, chunk %4d/%d, "
                "device %d [%d sources]", j+1, num_vis_dumps, chunk_index+1,
                num_sky_chunks, device_id, local_sky.num_sources);

        // Skip iteration if no sources above horizon.
        if (err == OSKAR_ERR_NO_VISIBLE_SOURCES) continue;
        else if (err != 0) return err;

        // Set dimensions of Jones matrices (this is not a resize!).
        err = J.set_size(n_stations, local_sky.num_sources); if (err) return err;
        err = R.set_size(n_stations, local_sky.num_sources); if (err) return err;
        err = E.set_size(n_stations, local_sky.num_sources); if (err) return err;
        err = K.set_size(n_stations, local_sky.num_sources); if (err) return err;

        // Average snapshot.
        for (int i = 0; i < num_vis_ave; ++i)
        {
            // Evaluate Greenwich Apparent Sidereal Time.
            double t_ave = t_dump + i * dt_ave;
            double gast = oskar_mjd_to_gast_fast(t_ave + dt_ave / 2);

            // Evaluate parallactic angle rotation (Jones R).
            err = oskar_evaluate_jones_R(&R, &local_sky, &tel_gpu, gast);
            if (err) return err;

            // Evaluate station beam (Jones E).
            err = oskar_evaluate_jones_E(&E, &local_sky, &tel_gpu, gast,
                    &work, &curand_state);
            if (err) return err;

            // Join Jones matrices (R = E * R).
            err = oskar_jones_join(&R, &E, &R);
            if (err) return err;

            for (int k = 0; k < num_fringe_ave; ++k)
            {
                // Evaluate Greenwich Apparent Sidereal Time.
                double t_fringe = t_ave + k * dt_fringe;
                double gast = oskar_mjd_to_gast_fast(t_fringe + dt_fringe / 2);

                // Evaluate station u,v,w coordinates.
                err = oskar_evaluate_station_uvw(&u, &v, &w, &tel_gpu, gast);
                if (err) return err;

                // Evaluate interferometer phase (Jones K).
                err = oskar_evaluate_jones_K(&K, &local_sky, &u, &v, &w);
                if (err) return err;

                // Join Jones matrices (J = K * R).
                err = oskar_jones_join(&J, &K, &R);
                if (err) return err;

                // Form baseline pairs.
                err = oskar_correlate(&vis, &J, &tel_gpu, &local_sky, &u, &v);
                if (err) return err;
            }
        }

        // Divide visibilities by number of averages.
        err = vis.scale_real(1.0 / (num_fringe_ave * num_vis_ave));
        if (err) return err;

        // Add visibilities to global data.
        err = vis_amp->insert(&vis, j * n_baselines);
        if (err) return err;
    }

    return OSKAR_SUCCESS;
}
