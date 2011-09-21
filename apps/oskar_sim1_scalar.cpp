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
#include "sky/oskar_date_time_to_mjd.h"

#include "station/oskar_StationModel.h"

#include "interferometry/oskar_TelescopeModel.h"
#include "interferometry/oskar_interferometer1_scalar.h"
#include "interferometry/oskar_VisData.h"

#include "apps/lib/oskar_load_telescope.h"
#include "apps/lib/oskar_load_stations.h"
#include "apps/lib/oskar_Settings.h"

#ifndef OSKAR_NO_MS
#include "apps/lib/oskar_write_ms.h"
#endif

#include "utility/oskar_cuda_device_info.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <math.h>

#include <QtCore/QTime>

using namespace std;

int sim1_d(const oskar_Settings& settings);
int sim1_f(const oskar_Settings& settings);


int main(int argc, char** argv)
{
    // $> oskar_sim1_scalar settings_file.txt
    if (argc != 2)
    {
        fprintf(stderr, "ERROR: Missing command line arguments.\n");
        fprintf(stderr, "Usage:  $ oskar_sim1_scalar [settings file]\n");
        return EXIT_FAILURE;
    }

    oskar_Settings settings;
    if (!settings.load(QString(argv[1]))) return EXIT_FAILURE;
    settings.print();

    QTime timer;
    timer.start();

    // Double precision.
    if (settings.double_precision())
    {
        sim1_d(settings);
    }

    // Single precision.
    else
    {
        sim1_f(settings);
    }

    printf("= Completed simulation after %f seconds.\n", timer.elapsed() / 1.0e3);

    return EXIT_SUCCESS;
}



int sim1_d(const oskar_Settings& settings)
{
    // ============== Load input data =========================================
    oskar_SkyModelGlobal_d sky;
    oskar_load_sources_d(settings.sky_file().toLatin1().data(), &sky);
    oskar_TelescopeModel_d telescope;
    oskar_load_telescope_d(settings.telescope_file().toLatin1().data(),
            settings.longitude_rad(), settings.latitude_rad(), &telescope);
    oskar_StationModel_d* stations;
    const char* station_dir = settings.station_dir().toLatin1().data();
    unsigned num_stations = oskar_load_stations_d(station_dir, &stations,
            &telescope.identical_stations);
    if (num_stations != telescope.num_antennas)
    {
        fprintf(stderr, "ERROR: Error loading telescope geometry.\n");
        return EXIT_FAILURE;
    }

    // ============== Simulation loop =========================================
    int error_code = 0;
    for (unsigned i = 0; i < settings.obs().num_channels(); ++i)
    {
        unsigned year   = settings.obs().start_time_utc_year();
        unsigned month  = settings.obs().start_time_utc_month();
        unsigned day    = settings.obs().start_time_utc_day();
        unsigned hour   = settings.obs().start_time_utc_hour();
        unsigned minute = settings.obs().start_time_utc_minute();
        double second   = settings.obs().start_time_utc_second();
        double day_fraction = (hour + minute/60 + second/3600) / 24.0;
        double start_time_mjd_utc = oskar_date_time_to_mjd_d(
                year, month, day, day_fraction);
        printf("- %i/%i/%i %i:%i:%f -> mjd %f\n", day, month, year, hour, minute,
                second, start_time_mjd_utc);

        double frequency = settings.obs().frequency(i);
        printf("- Frequency: %e\n", frequency);

        // Allocate memory for frequency scaled sky model.
        oskar_SkyModelGlobal_d sky_temp;
        sky_temp.num_sources = sky.num_sources;
        sky_temp.Dec = (double*) malloc(sky.num_sources * sizeof(double));
        sky_temp.RA  = (double*) malloc(sky.num_sources * sizeof(double));
        sky_temp.I   = (double*) malloc(sky.num_sources * sizeof(double));
        sky_temp.Q   = (double*) malloc(sky.num_sources * sizeof(double));
        sky_temp.U   = (double*) malloc(sky.num_sources * sizeof(double));
        sky_temp.V   = (double*) malloc(sky.num_sources * sizeof(double));
        memcpy(sky_temp.Dec, sky.Dec,  sky.num_sources * sizeof(double));
        memcpy(sky_temp.RA,  sky.RA,   sky.num_sources * sizeof(double));
        memcpy(sky_temp.I,   sky.I,    sky.num_sources * sizeof(double));
        for (int s = 0; s < sky.num_sources; ++s)
        {
//            sky_temp.I[s] = 1.0e6 * pow(frequency, -0.7);
            sky_temp.I[s] *= pow(frequency / settings.obs().start_frequency(), -0.7);
        }

        // Allocate visibility data.
        oskar_VisData_d vis;
        int num_baselines = num_stations * (num_stations-1) / 2;
        oskar_allocate_vis_data_d(num_baselines * settings.obs().num_vis_dumps(), &vis);

        error_code = oskar_interferometer1_scalar_d(telescope, stations, sky_temp,
                settings.obs().ra0_rad(), settings.obs().dec0_rad(),
                settings.obs().start_time_utc_mjd(), settings.obs().obs_length_days(),
                settings.obs().num_vis_dumps(), settings.obs().num_vis_ave(),
                settings.obs().num_fringe_ave(), frequency,
                settings.obs().channel_bandwidth(),
                settings.disable_station_beam(), &vis);

        printf("= Number of visibility points generated: %i\n", vis.num_samples);

        // Write visibility binary file.
        if (!settings.obs().oskar_vis_filename().isEmpty())
        {
            QString vis_file = settings.obs().oskar_vis_filename() + "_channel_" + QString::number(i) + ".dat";
            printf("= Writing OSKAR visibility data file: %s\n",
                    vis_file.toLatin1().data());
            oskar_write_vis_data_d(vis_file.toLatin1().data(), &vis);
        }

        // Write MS.
#ifndef OSKAR_NO_MS
        if (!settings.obs().ms_filename().isEmpty())
        {
            QString ms_file = settings.obs().ms_filename() + "_channel_" + QString::number(i) + ".ms";
            printf("= Writing Measurement Set: %s\n", ms_file.toLatin1().data());
            oskar_write_ms_d(ms_file.toLatin1().data(), &settings, &vis, i, true);
        }
#endif

        free(sky_temp.RA);
        free(sky_temp.Dec);
        free(sky_temp.I);
        free(sky_temp.Q);
        free(sky_temp.U);
        free(sky_temp.V);
        oskar_free_vis_data_d(&vis);
    }

    // ============== Cleanup =================================================
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
    free(stations);

    return EXIT_SUCCESS;
}




int sim1_f(const oskar_Settings& settings)
{
    // ============== Load input data =========================================
    oskar_SkyModelGlobal_f sky;
    oskar_load_sources_f(settings.sky_file().toLatin1().data(), &sky);
    oskar_TelescopeModel_f telescope;
    oskar_load_telescope_f(settings.telescope_file().toLatin1().data(),
            settings.longitude_rad(), settings.latitude_rad(), &telescope);
    oskar_StationModel_f* stations;
    const char* station_dir = settings.station_dir().toLatin1().data();
    unsigned num_stations = oskar_load_stations_f(station_dir, &stations,
            &telescope.identical_stations);
    if (num_stations != telescope.num_antennas)
    {
        fprintf(stderr, "ERROR: Error loading telescope geometry.\n");
        return EXIT_FAILURE;
    }

    // ============== Simulation loop =========================================
    int error_code = 0;
    for (unsigned i = 0; i < settings.obs().num_channels(); ++i)
    {
        unsigned year   = settings.obs().start_time_utc_year();
        unsigned month  = settings.obs().start_time_utc_month();
        unsigned day    = settings.obs().start_time_utc_day();
        unsigned hour   = settings.obs().start_time_utc_hour();
        unsigned minute = settings.obs().start_time_utc_minute();
        float second   = settings.obs().start_time_utc_second();
        float day_fraction = (hour + minute/60 + second/3600) / 24.0;
        float start_time_mjd_utc = oskar_date_time_to_mjd_d(
                year, month, day, day_fraction);
        printf("- %i/%i/%i %i:%i:%f -> mjd %f\n", day, month, year, hour, minute,
                second, start_time_mjd_utc);

        float frequency = settings.obs().frequency(i);
        printf("- Frequency: %e\n", frequency);

        // Allocate memory for frequency scaled sky model.
        oskar_SkyModelGlobal_f sky_temp;
        sky_temp.num_sources = sky.num_sources;
        sky_temp.Dec = (float*) malloc(sky.num_sources * sizeof(float));
        sky_temp.RA  = (float*) malloc(sky.num_sources * sizeof(float));
        sky_temp.I   = (float*) malloc(sky.num_sources * sizeof(float));
        sky_temp.Q   = (float*) malloc(sky.num_sources * sizeof(float));
        sky_temp.U   = (float*) malloc(sky.num_sources * sizeof(float));
        sky_temp.V   = (float*) malloc(sky.num_sources * sizeof(float));
        memcpy(sky_temp.Dec, sky.Dec,  sky.num_sources * sizeof(float));
        memcpy(sky_temp.RA,  sky.RA,   sky.num_sources * sizeof(float));
        memcpy(sky_temp.I,   sky.I,    sky.num_sources * sizeof(float));
        for (int s = 0; s < sky.num_sources; ++s)
        {
//            sky_temp.I[s] = 1.0e6 * pow(frequency, -0.7);
            sky_temp.I[s] = pow(frequency / settings.obs().start_frequency(), -0.7f);
//            printf("freq = %f, I = %f\n", frequency, sky_temp.I[s]);
        }

        // Allocate visibility data.
        oskar_VisData_f vis;
        int num_baselines = num_stations * (num_stations-1) / 2;
        oskar_allocate_vis_data_f(num_baselines * settings.obs().num_vis_dumps(), &vis);

        error_code = oskar_interferometer1_scalar_f(telescope, stations, sky_temp,
                settings.obs().ra0_rad(), settings.obs().dec0_rad(),
                start_time_mjd_utc, settings.obs().obs_length_days(),
                settings.obs().num_vis_dumps(), settings.obs().num_vis_ave(),
                settings.obs().num_fringe_ave(), frequency, settings.obs().channel_bandwidth(),
                settings.disable_station_beam(), &vis);

        printf("= Number of visibility points generated: %i\n", vis.num_samples);

        // Write visibility binary file.
        if (!settings.obs().oskar_vis_filename().isEmpty())
        {
            QString vis_file = settings.obs().oskar_vis_filename() + "_channel_" + QString::number(i) + ".dat";
            printf("= Writing OSKAR visibility data file: %s\n",
                    vis_file.toLatin1().data());
            oskar_write_vis_data_f(vis_file.toLatin1().data(), &vis);
        }

        // Write MS.
#ifndef OSKAR_NO_MS
        if (!settings.obs().ms_filename().isEmpty())
        {
            QString ms_file = settings.obs().ms_filename() + "_channel_" + QString::number(i) + ".ms";
            printf("= Writing Measurement Set: %s\n", ms_file.toLatin1().data());
            oskar_write_ms_f(ms_file.toLatin1().data(), &settings, &vis, i, true);
        }
#endif

        free(sky_temp.RA);
        free(sky_temp.Dec);
        free(sky_temp.I);
        free(sky_temp.Q);
        free(sky_temp.U);
        free(sky_temp.V);
        oskar_free_vis_data_f(&vis);
    }

    // ============== Cleanup =================================================
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
    free(stations);

    return EXIT_SUCCESS;
}

