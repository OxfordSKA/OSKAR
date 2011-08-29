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

#include "sky/oskar_SkyModel.h"
#include "sky/oskar_load_sources.h"

#include "station/oskar_StationModel.h"

#include "interferometry/oskar_TelescopeModel.h"
#include "interferometry/oskar_interferometer1_scalar.h"

#include "apps/oskar_load_telescope.h"
#include "apps/oskar_load_stations.h"
#include "apps/oskar_Settings.h"
#include "apps/oskar_VisData.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <QtCore/QTime>

int main(int argc, char** argv)
{
    // $> oskar_sim1_scalar settings_file.txt
    if (argc != 5)
    {
        fprintf(stderr, "ERROR: Missing command line arguments.\n");
        fprintf(stderr, "Usage:  $ sim1_scalar [settings file] [start freq] [freq inc] [num freq steps]\n");
        return EXIT_FAILURE;
    }

    oskar_Settings settings;
    if (!settings.load(QString(argv[1]))) return EXIT_FAILURE;
    settings.print();

    double freq_start  = atof(argv[2]);
    double freq_inc    = atof(argv[3]);
    int num_freq_steps = atoi(argv[4]);
    printf("freq start     = %f\n", freq_start);
    printf("freq inc       = %f\n", freq_inc);
    printf("num freq steps = %i\n", num_freq_steps);


    // =========================================================================
    // Load sky model.
    oskar_SkyModelGlobal_d sky;
    oskar_load_sources(settings.sky_file().toLatin1().data(), &sky);

    // Load telescope layout.
    oskar_TelescopeModel telescope;
    oskar_load_telescope(settings.telescope_file().toLatin1().data(),
            settings.longitude_rad(), settings.latitude_rad(), &telescope);

    // Load station layouts.
    oskar_StationModel* stations;
    const char* station_dir = settings.station_dir().toLatin1().data();
    unsigned num_stations = oskar_load_stations(station_dir, &stations,
            &telescope.identical_stations);

    // Check load worked.
    if (num_stations != telescope.num_antennas)
    {
        fprintf(stderr, "ERROR: Error loading telescope geometry.\n");
        return EXIT_FAILURE;
    }
    // =========================================================================


    for (int i = 0; i < num_freq_steps; ++i)
    {
        double frequency = freq_start + i * freq_inc;
        printf("simulation with freq %e\n", frequency);

        oskar_SkyModelGlobal_d sky_temp;
        sky_temp.num_sources = sky.num_sources;
        sky_temp.Dec = (double*) malloc(sky.num_sources * sizeof(double));
        sky_temp.RA = (double*) malloc(sky.num_sources * sizeof(double));
        sky_temp.I = (double*) malloc(sky.num_sources * sizeof(double));
        sky_temp.Q = (double*) malloc(sky.num_sources * sizeof(double));
        sky_temp.U = (double*) malloc(sky.num_sources * sizeof(double));
        sky_temp.V = (double*) malloc(sky.num_sources * sizeof(double));
        memcpy(sky_temp.Dec, sky.Dec, sky.num_sources * sizeof(double));
        memcpy(sky_temp.RA,  sky.RA, sky.num_sources * sizeof(double));
        memcpy(sky_temp.I,   sky.I, sky.num_sources * sizeof(double));
        for (int s = 0; s < sky.num_sources; ++s)
        {
            sky_temp.I[s] = 1.0e6 * pow(frequency, -0.7);
//            printf("%i %f\n", s, sky_temp.I[s]);
        }

        oskar_VisData vis(num_stations, settings.num_vis_dumps());

        QTime timer;
        timer.start();
        int err = oskar_interferometer1_scalar_d(telescope, stations, sky_temp,
                settings.ra0_rad(),
                settings.dec0_rad(),
                settings.obs_start_mjc_utc(),
                settings.obs_length_days(),
                settings.num_vis_dumps(),
                settings.num_vis_ave(),
                settings.num_fringe_ave(),
                frequency,
                settings.channel_bandwidth(),
                settings.disable_station_beam(),
                vis.vis(),
                vis.u(),
                vis.v(),
                vis.w());

        printf("= Completed simulation (%i of %i) after %f seconds [error code: %i].\n",
                i+1, num_freq_steps, timer.elapsed() / 1.0e3, err);

        printf("= Number of visibility points generated: %i\n", vis.size());

        QString outfile = "freq_scan_test_f_" + QString::number(frequency)+".dat";
        vis.write(outfile.toLatin1().data());

        free(sky_temp.RA);
        free(sky_temp.Dec);
        free(sky_temp.I);
        free(sky_temp.Q);
        free(sky_temp.U);
        free(sky_temp.V);
    }

    // =========================================================================
    // Free memory.
    free(sky.RA);
    free(sky.Dec);
    free(sky.I);
    free(sky.Q);
    free(sky.U);
    free(sky.V);

    free(telescope.antenna_x);
    free(telescope.antenna_y);
    free(telescope.antenna_z);

    for (unsigned i = 0; i < num_stations; ++i)
    {
        free(stations[i].antenna_x);
        free(stations[i].antenna_y);
    }
    // =========================================================================

    return EXIT_SUCCESS;
}
