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

#include "ms/oskar_MeasurementSet.h"
#include "apps/lib/oskar_file_utils.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "interferometry/oskar_Visibilities.h"

#include <QtCore/QDir>

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
        const oskar_TelescopeModel* telescope, int overwrite)
{
    // NOTE Currently the telescope model is needed for pointing and antenna
    // positions.

    // Check if the ms path already exists and overwrite if specified.
    QDir dir;
    dir.setPath(QString(ms_path));
    if (dir.exists(dir.absolutePath()))
    {
        // Try to overwrite.
        if (overwrite)
        {
            if (!oskar_remove_dir(ms_path))
                return OSKAR_ERR_UNKNOWN;
        }
        // No overwrite specified and directory already exists.
        else
        {
            return OSKAR_ERR_UNKNOWN;
        }
    }

    if (telescope->num_stations * (telescope->num_stations - 1) / 2 != vis->num_baselines)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    int num_antennas   = telescope->num_stations;
    int num_baselines  = vis->num_baselines;
    int num_pols       = vis->num_polarisations();
    int num_channels   = vis->num_channels;
    int num_times      = vis->num_times;
    double dt_vis_dump = vis->time_inc_seconds;
    double t_start_sec = vis->time_start_mjd_utc * DAYS_2_SEC + (dt_vis_dump / 2);
    double ref_freq    = vis->freq_start_hz;
    double chan_width  = vis->freq_inc_hz;

    // Create an empty measurement set of the correct dimensions and meta-data.
    oskar_MeasurementSet ms;
    ms.create(ms_path);
    if (telescope->type() == OSKAR_DOUBLE)
    {
        ms.addAntennas(num_antennas, (const double*)telescope->station_x,
                (const double*)telescope->station_y, (const double*)telescope->station_z);
    }
    else if (telescope->type() == OSKAR_SINGLE)
    {
        ms.addAntennas(num_antennas, (const float*)telescope->station_x,
                (const float*)telescope->station_y, (const float*)telescope->station_z);
    }
    else
    {
        return OSKAR_ERR_BAD_DATA_TYPE;
    }
    ms.addField(telescope->ra0_rad, telescope->dec0_rad);
    ms.addPolarisation(num_pols);
    for (int i = 0; i < num_pols; ++i)
    {
        ms.addBand(i, num_channels, ref_freq, chan_width);
    }

    // Evaluate baseline index arrays.
    int* baseline_ant_1 = (int*)malloc(vis->num_coords() * sizeof(int));
    int* baseline_ant_2 = (int*)malloc(vis->num_coords() * sizeof(int));
    for (int idx = 0, t = 0; t < num_times; ++t)
    {
        for (int a1 = 0; a1 < num_antennas; ++a1)
        {
            for (int a2 = (a1 + 1); a2 < num_antennas; ++a2)
            {
                baseline_ant_1[idx] = a1;
                baseline_ant_2[idx] = a2;
                ++idx;
            }
        }
    }

    if (vis->amplitude.is_double())
    {
        double2* amp_tb = (double2*)malloc(num_channels * num_pols * sizeof(double2));
        for (int t = 0; t < num_times; ++t)
        {
            // TODO check this is the correct midpoint time of the simulation
            // vis dump.
            double vis_time = t_start_sec + (t * dt_vis_dump);

            for (int b = 0; b < num_baselines; ++b)
            {
                int row = t * num_baselines + b;
                double u = ((double*)vis->uu_metres.data)[row];
                double v = ((double*)vis->vv_metres.data)[row];
                double w = ((double*)vis->ww_metres.data)[row];

                int ant1 = baseline_ant_1[row];
                int ant2 = baseline_ant_2[row];

                // Construct the amplitude data for the given time, baseline.
                for (int c = 0; c < num_channels; ++c)
                {
                    if (num_pols == 1)
                    {
                        int idx = num_baselines * (c * num_times + t) + b;
                        double2* vis_amp = &((double2*)vis->amplitude.data)[idx];
                        // xx
                        amp_tb[c * num_pols + 0].x = vis_amp->x;
                        amp_tb[c * num_pols + 0].y = vis_amp->y;
                    }
                    else
                    {
                        int idx = num_baselines * (c * num_times + t) + b;
                        double4c* vis_amp = &((double4c*)vis->amplitude.data)[idx];
                        // xx
                        amp_tb[c * num_pols + 0].x = vis_amp->a.x;
                        amp_tb[c * num_pols + 0].y = vis_amp->a.y;
                        // xy
                        amp_tb[c * num_pols + 1].x = vis_amp->b.x;
                        amp_tb[c * num_pols + 1].y = vis_amp->b.y;
                        // yx
                        amp_tb[c * num_pols + 2].x = vis_amp->c.x;
                        amp_tb[c * num_pols + 2].y = vis_amp->c.y;
                        // yy
                        amp_tb[c * num_pols + 3].x = vis_amp->d.x;
                        amp_tb[c * num_pols + 3].y = vis_amp->d.y;
                    }
                }
                ms.addVisibilities(num_pols, num_channels, 1, &u, &v, &w,
                        (double*)amp_tb, &ant1, &ant2, dt_vis_dump, dt_vis_dump,
                        &vis_time);
            }
        }
        free(amp_tb);
    }
    else if (vis->amplitude.is_single())
    {
        float2* amp_tb = (float2*)malloc(num_channels * num_pols * sizeof(float2));
        for (int t = 0; t < num_times; ++t)
        {
            // TODO check this is the correct midpoint time of the simulation
            // vis dump.
            float vis_time = t_start_sec + (t * dt_vis_dump);

            for (int b = 0; b < num_baselines; ++b)
            {
                int row = t * num_baselines + b;
                float u = ((float*)vis->uu_metres.data)[row];
                float v = ((float*)vis->vv_metres.data)[row];
                float w = ((float*)vis->ww_metres.data)[row];
                int ant1 = baseline_ant_1[row];
                int ant2 = baseline_ant_2[row];

                // Construct the amplitude data for the given time, baseline.
                for (int c = 0; c < num_channels; ++c)
                {
                    if (num_pols == 1)
                    {
                        int idx = num_baselines * (c * num_times + t) + b;
                        float2* vis_amp = &((float2*)vis->amplitude.data)[idx];
                        // xx
                        amp_tb[c * num_pols + 0].x = vis_amp->x;
                        amp_tb[c * num_pols + 0].y = vis_amp->y;
                    }
                    else
                    {
                        int idx = num_baselines * (c * num_times + t) + b;
                        float4c* vis_amp = &((float4c*)vis->amplitude.data)[idx];
                        // xx
                        amp_tb[c * num_pols + 0].x = vis_amp->a.x;
                        amp_tb[c * num_pols + 0].y = vis_amp->a.y;
                        // xy
                        amp_tb[c * num_pols + 1].x = vis_amp->b.x;
                        amp_tb[c * num_pols + 1].y = vis_amp->b.y;
                        // yx
                        amp_tb[c * num_pols + 2].x = vis_amp->c.x;
                        amp_tb[c * num_pols + 2].y = vis_amp->c.y;
                        // yy
                        amp_tb[c * num_pols + 3].x = vis_amp->d.x;
                        amp_tb[c * num_pols + 3].y = vis_amp->d.y;
                    }
                }
                ms.addVisibilities(num_pols, num_channels, 1, &u, &v, &w,
                        (float*)amp_tb, &ant1, &ant2, dt_vis_dump, dt_vis_dump, &vis_time);
            }
        }
        free(amp_tb);
    }
    else
    {
        return OSKAR_ERR_BAD_DATA_TYPE;
    }

    // Cleanup.
    free(baseline_ant_1);
    free(baseline_ant_2);

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
