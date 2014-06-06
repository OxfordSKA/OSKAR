/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include "apps/lib/oskar_vis_write_ms.h"
#include "apps/lib/oskar_remove_dir.h"
#include <oskar_vis.h>

#include "ms/oskar_MeasurementSet.h"

#include <QtCore/QDir>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef DAYS_2_SEC
#define DAYS_2_SEC 86400.0
#endif

extern "C"
void oskar_vis_write_ms(const oskar_Vis* vis,
        const char* ms_path, int overwrite, const char* run_log,
        size_t run_log_length, int* status)
{
    // Check all inputs.
    if (!vis || !ms_path || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    // Check if safe to proceed.
    if (*status) return;

    // Check if the Measurement Set already exists, and overwrite if specified.
    QDir dir;
    dir.setPath(QString(ms_path));
    if (dir.exists(dir.absolutePath()))
    {
        // Try to overwrite.
        if (overwrite)
        {
            if (!oskar_remove_dir(ms_path))
            {
                *status = OSKAR_ERR_FILE_IO;
                return;
            }
        }
        // No overwrite specified and directory already exists.
        else
        {
            *status = OSKAR_ERR_UNKNOWN;
            return;
        }
    }

    // Get dimensions.
    int num_antennas   = oskar_vis_num_stations(vis);
    int num_baselines  = oskar_vis_num_baselines(vis);
    int num_pols       = oskar_vis_num_polarisations(vis);
    int num_channels   = oskar_vis_num_channels(vis);
    int num_times      = oskar_vis_num_times(vis);
    int num_coords     = num_times * num_baselines;
    double dt_vis_dump = oskar_vis_time_inc_sec(vis);
    double t_start_sec = oskar_vis_time_start_mjd_utc(vis) * DAYS_2_SEC +
            (dt_vis_dump / 2);
    double ref_freq    = oskar_vis_freq_start_hz(vis);
    /* NOTE should the chan_width == freq_inc or channel bandwidth...
     * these are in general different numbers */
    double chan_width  = oskar_vis_freq_inc_hz(vis);
    if (num_antennas * (num_antennas - 1) / 2 != num_baselines)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    // Set channel width to be greater than 0, if it isn't already.
    // This is required for the Measurement Set to be valid.
    if (! (chan_width > 0.0))
        chan_width = 1.0;

    // Create an empty Measurement Set.
    oskar_MeasurementSet ms;
    ms.create(ms_path);

    // Add the antenna positions.
    const oskar_Mem* x_metres = oskar_vis_station_x_offset_ecef_metres_const(vis);
    const oskar_Mem* y_metres = oskar_vis_station_y_offset_ecef_metres_const(vis);
    const oskar_Mem* z_metres = oskar_vis_station_z_offset_ecef_metres_const(vis);
    if (oskar_mem_type(x_metres) == OSKAR_DOUBLE)
    {
        ms.addAntennas(num_antennas, oskar_mem_double_const(x_metres, status),
                oskar_mem_double_const(y_metres, status),
                oskar_mem_double_const(z_metres, status));
    }
    else if (oskar_mem_type(x_metres) == OSKAR_SINGLE)
    {
        ms.addAntennas(num_antennas, oskar_mem_float_const(x_metres, status),
                oskar_mem_float_const(y_metres, status),
                oskar_mem_float_const(z_metres, status));
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    // Add the field data.
    ms.addField(oskar_vis_phase_centre_ra_deg(vis) * M_PI / 180,
            oskar_vis_phase_centre_dec_deg(vis) * M_PI / 180);

    // Add polarisation and channel data.
    ms.addPolarisation(num_pols);
    ms.addBand(0, num_channels, ref_freq, chan_width);

    // Evaluate baseline index arrays.
    int* baseline_ant_1 = (int*)malloc(num_coords * sizeof(int));
    int* baseline_ant_2 = (int*)malloc(num_coords * sizeof(int));
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

    // Add visibilities and u,v,w coordinates.
    const oskar_Mem* amp = oskar_vis_amplitude_const(vis);
    const oskar_Mem* uu = oskar_vis_baseline_uu_metres_const(vis);
    const oskar_Mem* vv = oskar_vis_baseline_vv_metres_const(vis);
    const oskar_Mem* ww = oskar_vis_baseline_ww_metres_const(vis);
    const void* amp_ = oskar_mem_void_const(amp);
    if (oskar_mem_is_double(amp))
    {
        double2* amp_tb = (double2*)malloc(num_channels * num_pols * sizeof(double2));
        const double *uu_, *vv_, *ww_;
        uu_ = oskar_mem_double_const(uu, status);
        vv_ = oskar_mem_double_const(vv, status);
        ww_ = oskar_mem_double_const(ww, status);
        for (int t = 0; t < num_times; ++t)
        {
            double vis_time = t_start_sec + (t * dt_vis_dump);

            for (int b = 0; b < num_baselines; ++b)
            {
                int row = t * num_baselines + b;
                double u = uu_[row];
                double v = vv_[row];
                double w = ww_[row];

                int ant1 = baseline_ant_1[row];
                int ant2 = baseline_ant_2[row];

                // Construct the amplitude data for the given time, baseline.
                for (int c = 0; c < num_channels; ++c)
                {
                    int idx = num_baselines * (c * num_times + t) + b;
                    int i_out = c * num_pols;
                    if (num_pols == 1)
                    {
                        double2 vis_amp = ((const double2*)amp_)[idx];
                        amp_tb[i_out + 0] = vis_amp; // XX
                    }
                    else
                    {
                        double4c vis_amp = ((const double4c*)amp_)[idx];
                        amp_tb[i_out + 0] = vis_amp.a; // XX
                        amp_tb[i_out + 1] = vis_amp.b; // XY
                        amp_tb[i_out + 2] = vis_amp.c; // YX
                        amp_tb[i_out + 3] = vis_amp.d; // YY
                    }
                }
                ms.addVisibilities(num_pols, num_channels, 1, &u, &v, &w,
                        (double*)amp_tb, &ant1, &ant2, dt_vis_dump, dt_vis_dump,
                        &vis_time);
            }
        }
        free(amp_tb);
    }
    else if (oskar_mem_is_single(amp))
    {
        float2* amp_tb = (float2*)malloc(num_channels * num_pols * sizeof(float2));
        const float *uu_, *vv_, *ww_;
        uu_ = oskar_mem_float_const(uu, status);
        vv_ = oskar_mem_float_const(vv, status);
        ww_ = oskar_mem_float_const(ww, status);
        for (int t = 0; t < num_times; ++t)
        {
            float vis_time = t_start_sec + (t * dt_vis_dump);

            for (int b = 0; b < num_baselines; ++b)
            {
                int row = t * num_baselines + b;
                float u = uu_[row];
                float v = vv_[row];
                float w = ww_[row];
                int ant1 = baseline_ant_1[row];
                int ant2 = baseline_ant_2[row];

                // Construct the amplitude data for the given time, baseline.
                for (int c = 0; c < num_channels; ++c)
                {
                    int idx = num_baselines * (c * num_times + t) + b;
                    int i_out = c * num_pols;
                    if (num_pols == 1)
                    {
                        float2 vis_amp = ((const float2*)amp_)[idx];
                        amp_tb[i_out + 0] = vis_amp; // XX
                    }
                    else
                    {
                        float4c vis_amp = ((const float4c*)amp_)[idx];
                        amp_tb[i_out + 0] = vis_amp.a; // XX
                        amp_tb[i_out + 1] = vis_amp.b; // XY
                        amp_tb[i_out + 2] = vis_amp.c; // YX
                        amp_tb[i_out + 3] = vis_amp.d; // YY
                    }
                }
                ms.addVisibilities(num_pols, num_channels, 1, &u, &v, &w,
                        (float*)amp_tb, &ant1, &ant2, dt_vis_dump, dt_vis_dump,
                        &vis_time);
            }
        }
        free(amp_tb);
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }

    // Add the settings.
    ms.addSettings(oskar_mem_char_const(oskar_vis_settings_const(vis)),
            oskar_mem_length(oskar_vis_settings_const(vis)));

    // Add the run log.
    ms.addLog(run_log, run_log_length);

    // Cleanup.
    free(baseline_ant_1);
    free(baseline_ant_2);
}
