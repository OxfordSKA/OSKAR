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
#include "interferometry/oskar_evaluate_baselines.h"
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
#include <QtCore/QByteArray>
#include <QtCore/QTime>

oskar_TelescopeModel* oskar_set_up_telescope(const oskar_Settings& settings);
oskar_SkyModel* oskar_set_up_sky(const oskar_Settings& settings);

int main(int argc, char** argv)
{
    // Parse command line.
    int err = 0;
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

    QTime timer;
    timer.start();

    // Get the sky model and telescope model.
    oskar_SkyModel *sky_cpu, *sky_gpu;
    oskar_TelescopeModel *tel_cpu, *tel_gpu;
    sky_cpu = oskar_set_up_sky(settings);
    tel_cpu = oskar_set_up_telescope(settings);

    // Copy sky and telescope models to GPU.
    sky_gpu = new oskar_SkyModel(sky_cpu, OSKAR_LOCATION_GPU);
    tel_gpu = new oskar_TelescopeModel(tel_cpu, OSKAR_LOCATION_GPU);

    // Scale GPU telescope coordinates by wavenumber (get freq of channel 0).
    err = tel_gpu->multiply_by_wavenumber(settings.obs().frequency(0));
    if (err) oskar_exit(err);

    // Initialise blocks of Jones matrices and visibilities.
    int n_stations = tel_gpu->num_stations;
    int n_baselines = n_stations * (n_stations - 1) / 2;
    int n_sources = sky_gpu->num_sources;
    int complex_scalar = type | OSKAR_COMPLEX;
    int complex_matrix = type | OSKAR_COMPLEX | OSKAR_MATRIX;
    oskar_Jones J(complex_matrix, OSKAR_LOCATION_GPU, n_stations, n_sources);
    oskar_Jones R(complex_matrix, OSKAR_LOCATION_GPU, n_stations, n_sources);
    oskar_Jones E(complex_scalar, OSKAR_LOCATION_GPU, n_stations, n_sources);
    oskar_Jones K(complex_scalar, OSKAR_LOCATION_GPU, n_stations, n_sources);
    oskar_Mem vis(complex_matrix, OSKAR_LOCATION_GPU, n_baselines);
    oskar_Mem u(type, OSKAR_LOCATION_GPU, n_stations, true);
    oskar_Mem v(type, OSKAR_LOCATION_GPU, n_stations, true);
    oskar_Mem w(type, OSKAR_LOCATION_GPU, n_stations, true);
    oskar_Mem u_cpu(type, OSKAR_LOCATION_CPU, n_stations, true);
    oskar_Mem v_cpu(type, OSKAR_LOCATION_CPU, n_stations, true);
    oskar_Mem w_cpu(type, OSKAR_LOCATION_CPU, n_stations, true);
    oskar_Work work(type, OSKAR_LOCATION_GPU);
    oskar_Mem bu, bv, bw; // Pointers.

    // Calculate time increments.
    int num_vis_dumps        = settings.obs().num_vis_dumps();
    int num_vis_ave          = settings.obs().num_vis_ave();
    int num_fringe_ave       = settings.obs().num_fringe_ave();
    double obs_start_mjd_utc = settings.obs().start_time_utc_mjd();
    double obs_length        = settings.obs().obs_length_days();
    double dt_dump           = obs_length / num_vis_dumps; // Dump interval.
    double dt_ave            = dt_dump / num_vis_ave; // Average interval.
    double dt_fringe         = dt_ave / num_fringe_ave; // Fringe interval.

    // Create the global visibility structure on the CPU.
    oskar_Visibilities vis_global(complex_matrix, OSKAR_LOCATION_CPU,
            num_vis_dumps, n_baselines, 1);

    // Start simulation.
    for (int j = 0; j < num_vis_dumps; ++j)
    {
        // Start time for the visibility dump, in MJD(UTC).
        printf("--> Simulating snapshot (%i / %i).\n", j+1, num_vis_dumps);
        double t_dump = obs_start_mjd_utc + j * dt_dump;

        // Initialise visibilities for the dump to zero.
        err = vis.clear_contents();
        if (err) oskar_exit(err);

        // Average snapshot.
        for (int i = 0; i < num_vis_ave; ++i)
        {
            // Evaluate Greenwich Apparent Sidereal Time.
            double t_ave = t_dump + i * dt_ave;
            double gast = oskar_mjd_to_gast_fast(t_ave + dt_ave / 2);

            // Evaluate parallactic angle rotation (Jones R).
            err = oskar_evaluate_jones_R(&R, sky_gpu, tel_gpu, gast);
            if (err) oskar_exit(err);

            // Evaluate station beam (Jones E).
            err = oskar_evaluate_jones_E(&E, sky_gpu, tel_gpu, gast, &work);
            if (err) oskar_exit(err);

            // Join Jones matrices (R = E * R).
            err = oskar_jones_join(&R, &E, &R);
            if (err) oskar_exit(err);

            for (int k = 0; k < num_fringe_ave; ++k)
            {
                // Evaluate Greenwich Apparent Sidereal Time.
                double t_fringe = t_ave + k * dt_fringe;
                double gast = oskar_mjd_to_gast_fast(t_fringe + dt_fringe / 2);

                // Evaluate station u,v,w coordinates.
                err = oskar_evaluate_station_uvw(&u, &v, &w, tel_gpu, gast);
                if (err) oskar_exit(err);

                // Evaluate interferometer phase (Jones K).
                err = oskar_evaluate_jones_K(&K, sky_gpu, &u, &v, &w);
                if (err) oskar_exit(err);

                // Join Jones matrices (J = K * R).
                err = oskar_jones_join(&J, &K, &R);
                if (err) oskar_exit(err);

                // Produce visibilities.
                err = oskar_correlate(&vis, &J, tel_gpu, sky_gpu, &u, &v);
                if (err) oskar_exit(err);
            }

            // TODO Divide visibilities by number of fringe averages.
        }

        // TODO Divide visibilities by number of time averages.

        // Compute u,v,w coordinates of mid point.
        double gast = oskar_mjd_to_gast_fast(t_dump + dt_dump / 2.0);
        err = oskar_evaluate_station_uvw(&u_cpu, &v_cpu, &w_cpu, tel_cpu, gast);
        if (err) oskar_exit(err);

        // Extract pointers to baseline u,v,w coordinates for this dump.
        bu = vis_global.baseline_u.get_pointer(j * n_baselines, n_baselines);
        bv = vis_global.baseline_v.get_pointer(j * n_baselines, n_baselines);
        bw = vis_global.baseline_w.get_pointer(j * n_baselines, n_baselines);
        if (bu.data == NULL || bv.data == NULL || bw.data == NULL)
            oskar_exit(OSKAR_ERR_UNKNOWN);

        // Compute baselines from station positions.
        err = oskar_evaluate_baselines(&bu, &bv, &bw, &u_cpu, &v_cpu, &w_cpu);
        if (err) oskar_exit(err);

        // Add to global data.
        err = vis_global.amplitude.insert(&vis, j * n_baselines);
        if (err) oskar_exit(err);
    }

    // Write global visibilities to disk.
    if (!settings.obs().oskar_vis_filename().isEmpty())
    {
        QByteArray outname = settings.obs().oskar_vis_filename().toAscii();
        printf("--> Writing visibility file: '%s'\n", outname.constData());
        err = vis_global.write(outname);
        if (err) oskar_exit(err);
    }

    // Delete data structures.
    delete sky_cpu;
    delete sky_gpu;
    delete tel_gpu;
    delete tel_cpu;

    printf("=== Completed simulation after %f seconds.\n", timer.elapsed() / 1.0e3);

    return EXIT_SUCCESS;
}

oskar_SkyModel* oskar_set_up_sky(const oskar_Settings& settings)
{
    // Load sky model into CPU structure.
    QByteArray sky_file = settings.sky_file().toAscii();
    int type = settings.double_precision() ? OSKAR_DOUBLE : OSKAR_SINGLE;
    oskar_SkyModel *sky = new oskar_SkyModel(type, OSKAR_LOCATION_CPU);
    int err = sky->load(sky_file.constData());
    if (err) oskar_exit(err);

    // Compute source direction cosines relative to phase centre.
    err = sky->compute_relative_lmn(settings.obs().ra0_rad(),
            settings.obs().dec0_rad());
    if (err) oskar_exit(err);

    // Print summary data.
    printf("\n");
    printf("= Sky (%s)\n", sky_file.constData());
    printf("  - Num. sources           = %u\n", sky->num_sources);
    printf("\n");

    // Return the structure.
    return sky;
}

oskar_TelescopeModel* oskar_set_up_telescope(const oskar_Settings& settings)
{
    // Load telescope model into CPU structure.
    oskar_TelescopeModel *telescope;
    QByteArray telescope_file = settings.telescope_file().toAscii();
    QByteArray station_dir = settings.station_dir().toAscii();
    int type = settings.double_precision() ? OSKAR_DOUBLE : OSKAR_SINGLE;
    telescope = new oskar_TelescopeModel(type, OSKAR_LOCATION_CPU);
    int err = telescope->load_station_pos(telescope_file.constData(),
            settings.longitude_rad(), settings.latitude_rad(),
            settings.altitude_m());
    if (err) oskar_exit(err);

    // Load stations from directory.
    err = oskar_load_stations(telescope->station,
            &(telescope->identical_stations), telescope->num_stations,
            station_dir.constData());
    if (err) oskar_exit(err);

    // Set phase centre.
    telescope->ra0 = settings.obs().ra0_rad();
    telescope->dec0 = settings.obs().dec0_rad();

    // Set other telescope parameters.
    telescope->use_common_sky = true; // FIXME set this via the settings file.

    // Set other station parameters.
    for (int i = 0; i < telescope->num_stations; ++i)
    {
        telescope->station[i].ra0 = telescope->ra0;
        telescope->station[i].dec0 = telescope->dec0;
        telescope->station[i].single_element_model = true; // FIXME set this via the settings file.
    }

    // Print summary data.
    printf("\n");
    printf("= Telescope (%s)\n", telescope_file.constData());
    printf("  - Num. stations          = %u\n", telescope->num_stations);
    printf("  - Identical stations     = %s\n",
            telescope->identical_stations ? "true" : "false");
    printf("\n");

    // Return the structure.
    return telescope;
}
