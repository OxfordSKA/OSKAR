/*
 * Copyright (c) 2011-2014, The University of Oxford
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
#include <oskar_Dir.h>

#include "ms/oskar_MeasurementSet.h"

#include <cmath>
#include <cstdlib>

extern "C"
void oskar_vis_write_ms(const oskar_Vis* vis, const char* ms_path,
        int overwrite, const char* run_log, size_t run_log_length, int* status)
{
    const oskar_Mem *x_metres, *y_metres, *z_metres, *amp, *uu, *vv, *ww;
    double dt_dump, t_start_sec, ref_freq, chan_width;
    int num_antennas, num_baselines, num_pols, num_channels, num_times;
    int precision;

    // Check all inputs.
    if (!vis || !ms_path || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    // Check if safe to proceed.
    if (*status) return;

    // Check if the Measurement Set already exists, and overwrite if specified.
    oskar_Dir dir(ms_path);
    if (dir.exists())
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

    // Pull data from visibility structure.
    num_antennas  = oskar_vis_num_stations(vis);
    num_baselines = oskar_vis_num_baselines(vis);
    num_pols      = oskar_vis_num_polarisations(vis);
    num_channels  = oskar_vis_num_channels(vis);
    num_times     = oskar_vis_num_times(vis);
    dt_dump       = oskar_vis_time_inc_sec(vis);
    t_start_sec   = oskar_vis_time_start_mjd_utc(vis) * 86400 + (dt_dump / 2);
    ref_freq      = oskar_vis_freq_start_hz(vis);
    /* NOTE Should the chan_width be freq_inc or channel bandwidth?
     * (These are different numbers in general.) */
    chan_width    = oskar_vis_freq_inc_hz(vis);
    x_metres      = oskar_vis_station_x_offset_ecef_metres_const(vis);
    y_metres      = oskar_vis_station_y_offset_ecef_metres_const(vis);
    z_metres      = oskar_vis_station_z_offset_ecef_metres_const(vis);
    amp           = oskar_vis_amplitude_const(vis);
    uu            = oskar_vis_baseline_uu_metres_const(vis);
    vv            = oskar_vis_baseline_vv_metres_const(vis);
    ww            = oskar_vis_baseline_ww_metres_const(vis);
    precision     = oskar_mem_precision(amp);

    // Set channel width to be greater than 0, if it isn't already.
    // This is required for the Measurement Set to be valid.
    if (! (chan_width > 0.0))
        chan_width = 1.0;

    // Create an empty Measurement Set.
    oskar_MeasurementSet ms;
    ms.create(ms_path);

    // Add the antenna positions.
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

    // Add the field, polarisation and channel data.
    ms.addField(oskar_vis_phase_centre_ra_deg(vis) * M_PI / 180,
            oskar_vis_phase_centre_dec_deg(vis) * M_PI / 180);
    ms.addPolarisation(num_pols);
    ms.addBand(0, num_channels, ref_freq, chan_width);

    // Evaluate baseline index arrays.
    int* baseline_ant_1 = (int*)malloc(num_baselines * sizeof(int));
    int* baseline_ant_2 = (int*)malloc(num_baselines * sizeof(int));
    for (int a1 = 0, i = 0; a1 < num_antennas; ++a1)
    {
        for (int a2 = a1 + 1; a2 < num_antennas; ++a2, ++i)
        {
            baseline_ant_1[i] = a1;
            baseline_ant_2[i] = a2;
        }
    }

    // Add visibilities and u,v,w coordinates.
    const void* amp_ = oskar_mem_void_const(amp);
    oskar_Mem* amp_tb_ = oskar_mem_create(precision | OSKAR_COMPLEX, OSKAR_CPU,
            num_baselines * num_channels * num_pols, status);
    oskar_Mem* times = oskar_mem_create(precision, OSKAR_CPU, num_baselines,
            status);
    if (precision == OSKAR_DOUBLE)
    {
        double2* amp_tb;
        const double *uu_, *vv_, *ww_;
        uu_ = oskar_mem_double_const(uu, status);
        vv_ = oskar_mem_double_const(vv, status);
        ww_ = oskar_mem_double_const(ww, status);
        amp_tb = oskar_mem_double2(amp_tb_, status);
        for (int t = 0; t < num_times; ++t)
        {
            int row = t * num_baselines;
            double vis_time = t_start_sec + (t * dt_dump);
            oskar_mem_set_value_real(times, vis_time, 0, num_baselines, status);

            for (int b = 0; b < num_baselines; ++b)
            {
                // Construct the amplitude data for the given time, baseline.
                for (int c = 0; c < num_channels; ++c)
                {
                    int i_in = num_baselines * (num_times * c + t) + b;
                    int i_out = num_pols * (num_channels * b + c);
                    if (num_pols == 1)
                    {
                        amp_tb[i_out] = ((const double2*)amp_)[i_in]; // I
                    }
                    else
                    {
                        double4c vis_amp = ((const double4c*)amp_)[i_in];
                        amp_tb[i_out + 0] = vis_amp.a; // XX
                        amp_tb[i_out + 1] = vis_amp.b; // XY
                        amp_tb[i_out + 2] = vis_amp.c; // YX
                        amp_tb[i_out + 3] = vis_amp.d; // YY
                    }
                }
            }
            ms.addVisibilities(num_pols, num_channels, num_baselines,
                    &uu_[row], &vv_[row], &ww_[row], (const double*)amp_tb,
                    baseline_ant_1, baseline_ant_2, dt_dump, dt_dump,
                    oskar_mem_double_const(times, status));
        }
    }
    else if (precision == OSKAR_SINGLE)
    {
        float2* amp_tb;
        const float *uu_, *vv_, *ww_;
        uu_ = oskar_mem_float_const(uu, status);
        vv_ = oskar_mem_float_const(vv, status);
        ww_ = oskar_mem_float_const(ww, status);
        amp_tb = oskar_mem_float2(amp_tb_, status);
        for (int t = 0; t < num_times; ++t)
        {
            int row = t * num_baselines;
            double vis_time = t_start_sec + (t * dt_dump);
            oskar_mem_set_value_real(times, vis_time, 0, num_baselines, status);

            for (int b = 0; b < num_baselines; ++b)
            {
                // Construct the amplitude data for the given time, baseline.
                for (int c = 0; c < num_channels; ++c)
                {
                    int i_in = num_baselines * (num_times * c + t) + b;
                    int i_out = num_pols * (num_channels * b + c);
                    if (num_pols == 1)
                    {
                        amp_tb[i_out] = ((const float2*)amp_)[i_in]; // I
                    }
                    else
                    {
                        float4c vis_amp = ((const float4c*)amp_)[i_in];
                        amp_tb[i_out + 0] = vis_amp.a; // XX
                        amp_tb[i_out + 1] = vis_amp.b; // XY
                        amp_tb[i_out + 2] = vis_amp.c; // YX
                        amp_tb[i_out + 3] = vis_amp.d; // YY
                    }
                }
            }
            ms.addVisibilities(num_pols, num_channels, num_baselines,
                    &uu_[row], &vv_[row], &ww_[row], (const float*)amp_tb,
                    baseline_ant_1, baseline_ant_2, dt_dump, dt_dump,
                    oskar_mem_float_const(times, status));
        }
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
    oskar_mem_free(amp_tb_, status);
    oskar_mem_free(times, status);
    free(baseline_ant_1);
    free(baseline_ant_2);
}
