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

#include "apps/lib/oskar_write_ms.h"
#include "apps/lib/oskar_load_telescope.h"
#include "apps/lib/oskar_file_utils.h"
#include "ms/oskar_ms.h"
#include "interferometry/oskar_TelescopeModel.h"

#include "interferometry/oskar_Visibilities.h"
#include "interferometry/oskar_SimTime.h"
#include "ms/oskar_MeasurementSet.h"

#include <QtCore/QDir>
#include <QtCore/QFileInfoList>
#include <QtCore/QFile>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef DAYS_2_SEC
#define DAYS_2_SEC 86400.0
#endif

#ifdef __cplusplus
extern "C" {
#endif

int oskar_write_ms(const char* ms_path, const oskar_Visibilities* vis,
        const oskar_Settings* settings, int overwrite)
{
    // TODO
    // - Remove settings structure from the interface?
    //   - all of the relevant meta data should probably be in the vis structure.
    //   - This would also allow for other functions that operate on visibility
    //     data (e.g. imagers) to be not rely on the user remembering which
    //     settings file was used to generate the data.

    // Check if the ms path already exists and overwrite if specified.
    QDir dir;
    dir.setPath(QString(ms_path));
    if (dir.exists(dir.absolutePath()))
    {
        // Try to overwrite.
        if (overwrite && !oskar_remove_dir(ms_path))
            return OSKAR_ERR_UNKNOWN;
        // No overwrite specified and directory already exists.
        else
            return OSKAR_ERR_UNKNOWN;
    }

    // Load telescope model station / antenna positions.
    oskar_TelescopeModel telescope(OSKAR_DOUBLE, OSKAR_LOCATION_CPU);
    QByteArray telescope_file = settings->telescope_file().toAscii();
    int error = telescope.load_station_pos(telescope_file.constData(),
            settings->longitude_rad(), settings->latitude_rad(),
            settings->altitude_m());
    if (error) return error;

    // Create an empty measurement set of the correct dimensions and meta-data.
    oskar_MeasurementSet ms;
    ms.create(ms_path);
    ms.addAntennas(telescope.num_stations, telescope.station_x,
            telescope.station_y, telescope.station_z);
    ms.addField(settings->obs().ra0_rad(), settings->obs().dec0_rad());
    ms.addPolarisation(vis->num_polarisations());
    double ref_freq   = settings->obs().start_frequency();
    double chan_width = settings->obs().frequency_inc();
    for (int i = 0; i < vis->num_polarisations(); ++i)
    {
        ms.addBand(i, vis->num_channels, ref_freq, chan_width);
    }

    // Evaluate baseline index arrays.
    int* baseline_ant_1 = (int*)malloc(vis->num_coords() * sizeof(int));
    int* baseline_ant_2 = (int*)malloc(vis->num_coords() * sizeof(int));
    const oskar_SimTime* time = settings->obs().sim_time();
    for (int idx = 0, t = 0; t < time->num_vis_dumps; ++t)
    {
        for (int a1 = 0; a1 < (int)telescope.num_stations; ++a1)
        {
            for (int a2 = (a1 + 1); a2 < (int)telescope.num_stations; ++a2)
            {
                baseline_ant_1[idx] = a1;
                baseline_ant_2[idx] = a2;
                ++idx;
            }
        }
    }

    // Evaluate the time array.
    int num_baselines = telescope.num_stations * (telescope.num_stations - 1) / 2;
    if (num_baselines != vis->num_baselines)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    double interval = time->obs_length_seconds / time->num_vis_dumps;
    double* times = (double*)malloc(vis->num_coords() * sizeof(double));
    double t_start_sec = time->obs_start_mjd_utc * DAYS_2_SEC + interval / 2;
    for (int j = 0; j < time->num_vis_dumps; ++j)
    {
        double t = t_start_sec + interval * j;
        for (int i = 0; i < num_baselines; ++i)
        {
            times[j * num_baselines + i] = t;
        }
    }

    //ms.addVisibilities()



//    oskar_ms_append_vis1(ms_path, vis->num_samples, vis->u, vis->v, vis->w,
//            (double*)vis->amp, baseline_ant_1, baseline_ant_2,
//            exposure, interval, times);

    // Cleanup.
    free(baseline_ant_1);
    free(baseline_ant_2);
    free(times);

    return OSKAR_SUCCESS;
}























// DEPRECATED
void oskar_write_ms_d(const char* ms_path, const oskar_Settings* settings,
        const oskar_VisData_d* vis, const unsigned channel, const bool overwrite)
{
    const double days_to_sec = 86400.0;

    // Check if the specified ms path already exists and overwrite if specified.
    QDir dir;
    dir.setPath(QString(ms_path));
    if ( dir.exists(dir.absolutePath()) )
    {
        if (overwrite)
        {
            printf("WARNING: Existing measurement set '%s' will be removed.\n",
                    ms_path);
            if (!oskar_remove_dir(ms_path))
            {
                fprintf(stderr, "ERROR: Failed to remove existing MS!\n");
                return;
            }
        }
        else
        {
            fprintf(stderr, "ERROR: Measurement set '%s' already exists.",
                    ms_path);
            return;
        }
    }

    // Make local copies of settings.
    const oskar_SimTime* time = settings->obs().sim_time();
    double mjd_start = time->obs_start_mjd_utc;
    double ra0_rad   = settings->obs().ra0_rad();
    double dec0_rad  = settings->obs().dec0_rad();
    double frequency = settings->obs().start_frequency() + settings->obs().frequency_inc() * channel;

    // Load telescope model to get station/antenna positions.
    oskar_TelescopeModel_d telescope;
    oskar_load_telescope_d(settings->telescope_file().toLatin1().data(),
            settings->longitude_rad(), settings->latitude_rad(), &telescope);

    // Create the measurement set.
    oskar_ms_create_meta1(ms_path, ra0_rad, dec0_rad,
            telescope.num_antennas, telescope.antenna_x, telescope.antenna_y,
            telescope.antenna_z, frequency);

    // Evaluate baseline index arrays.
    int* baseline_ant_1 = (int*) malloc(vis->num_samples * sizeof(int));
    int* baseline_ant_2 = (int*) malloc(vis->num_samples * sizeof(int));
    for (int idx = 0, t = 0; t < time->num_vis_dumps; ++t)
    {
        for (int a1 = 0; a1 < (int)telescope.num_antennas; ++a1)
        {
            for (int a2 = (a1 + 1); a2 < (int)telescope.num_antennas; ++a2)
            {
                baseline_ant_1[idx] = a1;
                baseline_ant_2[idx] = a2;
                ++idx;
            }
        }
    }

    // Write visibility data to the measurement set.
    int num_baselines = telescope.num_antennas * (telescope.num_antennas - 1) / 2;
    double interval = time->obs_length_seconds / time->num_vis_dumps;
    double exposure = interval;
    double* times = (double*) malloc(vis->num_samples * sizeof(double));
    double t_start_sec = mjd_start * days_to_sec + interval / 2;
    for (int j = 0; j < time->num_vis_dumps; ++j)
    {
        double t = t_start_sec + interval * j;
        for (int i = 0; i < num_baselines; ++i)
            times[j * num_baselines + i] = t;
    }
    oskar_ms_append_vis1(ms_path, vis->num_samples, vis->u, vis->v, vis->w,
            (double*)vis->amp, baseline_ant_1, baseline_ant_2,
            exposure, interval, times);

   // Cleanup.
   free(baseline_ant_1);
   free(baseline_ant_2);
   free(times);
}

// DEPRECATED
void oskar_write_ms_f(const char* ms_path, const oskar_Settings* settings,
        const oskar_VisData_f* vis, const unsigned channel, const bool overwrite)
{
    oskar_VisData_d temp_vis;
    oskar_allocate_vis_data_d(vis->num_samples, &temp_vis);
    for (int i = 0; i < vis->num_samples; ++i)
    {
        temp_vis.u[i]     = (double)vis->u[i];
        temp_vis.v[i]     = (double)vis->v[i];
        temp_vis.w[i]     = (double)vis->w[i];
        temp_vis.amp[i].x = (double)vis->amp[i].x;
        temp_vis.amp[i].y = (double)vis->amp[i].y;
    }

    oskar_write_ms_d(ms_path, settings, &temp_vis, channel, overwrite);

    oskar_free_vis_data_d(&temp_vis);
}

#ifdef __cplusplus
}
#endif

