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

#include "apps/lib/oskar_load_stations.h"
#include "apps/lib/oskar_Settings.h"
//#include "apps/lib/oskar_set_up_sky.h"
//#include "apps/lib/oskar_set_up_telescope.h"
#include "interferometry/oskar_correlate.h"
#include "interferometry/oskar_evaluate_jones_K.h"
#include "interferometry/oskar_evaluate_station_uvw.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "interferometry/oskar_Visibilities.h"
#include "math/oskar_Jones.h"
#include "math/oskar_jones_join.h"
#include "sky/oskar_SkyModel.h"
#include "sky/oskar_evaluate_jones_R.h"
#include "sky/oskar_mjd_to_gast_fast.h"
#include "station/oskar_evaluate_jones_E.h"
#include "utility/oskar_exit.h"
#include "utility/oskar_Mem.h"

#include <cstdio>
#include <cstdlib>

oskar_TelescopeModel* oskar_set_up_telescope(const oskar_Settings& settings);
oskar_SkyModel* oskar_set_up_sky(const oskar_Settings& settings);

int main(int argc, char** argv)
{
    // Parse command line.
    if (argc != 2)
    {
        fprintf(stderr, "ERROR: Missing command line arguments.\n");
        fprintf(stderr, "Usage:  $ oskar_sim1 [settings file]\n");
        return EXIT_FAILURE;
    }

    // Load the settings file.
    oskar_Settings settings;
    if (!settings.load(QString(argv[1]))) return EXIT_FAILURE;
    settings.print();

    // Set the precision.
    int type = settings.double_precision() ? OSKAR_DOUBLE : OSKAR_SINGLE;

    // Get the sky model and telescope model.
    oskar_SkyModel* sky_gpu = oskar_set_up_sky(settings);
    oskar_TelescopeModel* telescope_gpu = oskar_set_up_telescope(settings);

    // Initialise blocks of Jones matrices and visibilities.
    int n_stat = telescope_gpu->num_stations;
    int n_src = sky_gpu->num_sources;
    int complex_scalar = type | OSKAR_COMPLEX;
    int complex_matrix = type | OSKAR_COMPLEX | OSKAR_MATRIX;
    oskar_Jones J(complex_matrix, OSKAR_LOCATION_GPU, n_stat, n_src);
    oskar_Jones R(complex_matrix, OSKAR_LOCATION_GPU, n_stat, n_src);
    oskar_Jones E(complex_scalar, OSKAR_LOCATION_GPU, n_stat, n_src);
    oskar_Jones K(complex_scalar, OSKAR_LOCATION_GPU, n_stat, n_src);
    oskar_Mem u(type, OSKAR_LOCATION_GPU, n_stat, true);
    oskar_Mem v(type, OSKAR_LOCATION_GPU, n_stat, true);
    oskar_Mem w(type, OSKAR_LOCATION_GPU, n_stat, true);
    oskar_Work work;
    oskar_Visibilities vis(complex_scalar, OSKAR_LOCATION_GPU);
    oskar_Visibilities vis_global(complex_scalar, OSKAR_LOCATION_CPU);

    // Calculate time increments.
    int num_vis_dumps        = settings.obs().num_vis_dumps();
    int num_vis_ave          = settings.obs().num_vis_ave();
    int num_fringe_ave       = settings.obs().num_fringe_ave();
    int total_samples        = num_vis_dumps * num_fringe_ave * num_vis_ave;
    double obs_start_mjd_utc = settings.obs().start_time_utc_mjd();
    double obs_length        = settings.obs().obs_length_days();
    double dt                = obs_length / total_samples; // Fringe interval.
    double dt_vis            = obs_length / num_vis_dumps; // Dump interval.
    double dt_vis_offset     = dt_vis / 2.0;
    double dt_vis_ave        = dt_vis / num_vis_ave; // Vis average interval.
    double dt_vis_ave_offset = dt_vis_ave / 2.0;

    // Start simulation.
    int err = 0;
    for (int j = 0; j < num_vis_dumps; ++j)
    {
        printf("--> Simulating snapshot (%i / %i).\n", j+1, num_vis_dumps);

        // Start time for the visibility dump, in MJD(UTC).
        double t_vis_dump_start = obs_start_mjd_utc + (j * dt_vis);

        // Initialise visibilities for the dump to zero.
        vis.clear_contents();

        // Average snapshot.
        for (int i = 0; i < num_vis_ave; ++i)
        {
            // Evaluate Greenwich Apparent Sidereal Time at mid-point.
            double t_ave_start = t_vis_dump_start + i * dt_vis_ave;
            double t_ave_mid   = t_ave_start + dt_vis_ave_offset;
            double gast = oskar_mjd_to_gast_fast(t_ave_mid);

            // Evaluate parallactic angle rotation (Jones R).
            err = oskar_evaluate_jones_R(&R, sky_gpu, telescope_gpu, gast);
            if (err) oskar_exit(err);

            // Evaluate station beam (Jones E).
            err = oskar_evaluate_jones_E(&E, sky_gpu, telescope_gpu, gast, &work);
            if (err) oskar_exit(err);

            // Join Jones matrices (R = E * R).
            err = oskar_jones_join(&R, &E, &R);
            if (err) oskar_exit(err);

            for (int k = 0; k < num_fringe_ave; ++k)
            {
                // Evaluate Greenwich Apparent Sidereal Time.
                double gast = oskar_mjd_to_gast_fast(t_ave_start + k * dt);

                // Evaluate station u,v,w coordinates.
                err = oskar_evaluate_station_uvw(&u, &v, &w, telescope_gpu, gast);
                if (err) oskar_exit(err);

                // Evaluate interferometer phase (Jones K).
                err = oskar_evaluate_jones_K(&K, sky_gpu, &u, &v, &w);
                if (err) oskar_exit(err);

                // Join Jones matrices (J = K * R).
                err = oskar_jones_join(&J, &K, &R);
                if (err) oskar_exit(err);

                // Produce visibilities.
                err = oskar_correlate(&vis, &J, telescope_gpu, sky_gpu, &u, &v);
                if (err) oskar_exit(err);
            }
        }

        // TODO Compute u,v,w coordinates of mid point.
//        err = oskar_evaluate_station_uvw(&u, &v, &w, telescope_gpu, gast);
//        if (err) oskar_exit(err);

        // Add to global data.
        vis_global.append(&vis);
    }

    // Write global visibilities to disk.
    vis_global.write(settings.obs().oskar_vis_filename().toLatin1().data());

    // Delete data structures on GPU.
    delete sky_gpu;
    delete telescope_gpu;

    return EXIT_SUCCESS;
}

oskar_SkyModel* oskar_set_up_sky(const oskar_Settings& settings)
{
    // Load sky model into CPU structure.
    oskar_SkyModel *sky_cpu, *sky_gpu;
    int type = settings.double_precision() ? OSKAR_DOUBLE : OSKAR_SINGLE;
    sky_cpu = new oskar_SkyModel(type, OSKAR_LOCATION_CPU);
    int err = sky_cpu->load(settings.sky_file().toLatin1().data());
    if (err) oskar_exit(err);

    // Copy sky model to GPU.
    sky_gpu = new oskar_SkyModel(sky_cpu, OSKAR_LOCATION_GPU);
    delete sky_cpu; sky_cpu = NULL;

    // Compute source direction cosines relative to phase centre.
    err = sky_gpu->compute_relative_lmn(settings.obs().ra0_rad(),
            settings.obs().dec0_rad());
    if (err) oskar_exit(err);

    // Return the structure.
    return sky_gpu;
}

oskar_TelescopeModel* oskar_set_up_telescope(const oskar_Settings& settings)
{
    // Load telescope model into CPU structure.
    oskar_TelescopeModel *telescope_cpu, *telescope_gpu;
    int type = settings.double_precision() ? OSKAR_DOUBLE : OSKAR_SINGLE;
    telescope_cpu = new oskar_TelescopeModel(type, OSKAR_LOCATION_CPU);
    int err = telescope_cpu->load_station_pos(
            settings.telescope_file().toLatin1().data(),
            settings.longitude_rad(), settings.latitude_rad(),
            settings.altitude_m());
    if (err) oskar_exit(err);

    // Load stations from directory.
    err = oskar_load_stations(telescope_cpu->station,
            &(telescope_cpu->identical_stations), telescope_cpu->num_stations,
            settings.station_dir().toLatin1().data());
    if (err) oskar_exit(err);

    // Copy telescope model to GPU.
    telescope_gpu = new oskar_TelescopeModel(telescope_cpu, OSKAR_LOCATION_GPU);
    delete telescope_cpu; telescope_cpu = NULL;

    // Return the structure.
    return telescope_gpu;
}
