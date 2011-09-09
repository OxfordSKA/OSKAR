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
#include "interferometry/oskar_VisData.h"

#include "apps/lib/oskar_load_telescope.h"
#include "apps/lib/oskar_load_stations.h"
#include "apps/lib/oskar_Settings.h"
#include "apps/lib/oskar_write_ms.h"

#include "utility/oskar_cuda_device_info.h"

#include <cstdio>
#include <cstdlib>

#include <QtCore/QTime>

int main(int argc, char** argv)
{
    // $> oskar_sim1_scalar settings_file.txt
    if (argc != 2)
    {
        fprintf(stderr, "ERROR: Missing command line arguments.\n");
        fprintf(stderr, "Usage:  $ sim1_scalar_f [settings file]\n");
        return EXIT_FAILURE;
    }

    oskar_Settings settings;
    if (!settings.load(QString(argv[1]))) return EXIT_FAILURE;
    settings.print();


    // =========================================================================
    // Load sky model.
    oskar_SkyModelGlobal_f sky;
    oskar_load_sources_f(settings.sky_file().toLatin1().data(), &sky);

    // Load telescope layout.
    oskar_TelescopeModel_f telescope;
    oskar_load_telescope_f(settings.telescope_file().toLatin1().data(),
            settings.longitude_rad(), settings.latitude_rad(), &telescope);

    // Load station layouts.
    oskar_StationModel_f* stations;
    const char* station_dir = settings.station_dir().toLatin1().data();
    unsigned num_stations = oskar_load_stations_f(station_dir, &stations,
            &telescope.identical_stations);

    // Check load worked.
    if (num_stations != telescope.num_antennas)
    {
        fprintf(stderr, "ERROR: Error loading telescope geometry.\n");
        return EXIT_FAILURE;
    }
    // =========================================================================

    oskar_VisData_f vis;
    int num_baselines = num_stations * (num_stations-1) / 2;
    oskar_allocate_vis_data_f(num_baselines * settings.num_vis_dumps(), &vis);

    QTime timer;
    timer.start();
    int err = oskar_interferometer1_scalar_f(telescope, stations, sky,
            settings.ra0_rad(),
            settings.dec0_rad(),
            settings.obs_start_mjd_utc(),
            settings.obs_length_days(),
            settings.num_vis_dumps(),
            settings.num_vis_ave(),
            settings.num_fringe_ave(),
            settings.frequency(),
            settings.channel_bandwidth(),
            settings.disable_station_beam(),
            &vis);

    printf("= Completed simulation after %f seconds [error code: %i].\n",
            timer.elapsed() / 1.0e3, err);

    printf("= Number of visibility points generated: %i\n", vis.num_samples);

    oskar_write_vis_data_f(settings.output_file().toLatin1().data(), &vis);

//    QString ms_file = settings.output_file() + ".ms";
//    oskar_VisData_d vis_d;
//    oskar_allocate_vis_data_d(vis.num_samples, &vis_d);
//    vis_d.num_samples = vis.num_samples;
//    for (int i = 0; i < vis.num_samples; ++i)
//    {
//        vis_d.u[i]     = (double)vis.u[i];
//        vis_d.v[i]     = (double)vis.v[i];
//        vis_d.w[i]     = (double)vis.w[i];
//        vis_d.amp[i].x = (double)vis.amp[i].x;
//        vis_d.amp[i].y = (double)vis.amp[i].y;
//    }
//    oskar_write_ms_d(ms_file.toLatin1().data(), &settings, &vis_d, true);
//    oskar_free_vis_data_d(&vis_d);

    // =========================================================================
    // Free memory.
    free(sky.RA);
    free(sky.Dec);
    free(sky.I);
    free(sky.Q);
    free(sky.U);
    free(sky.V);

    oskar_free_vis_data_f(&vis);

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
