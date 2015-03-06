/*
 * Copyright (c) 2011-2015, The University of Oxford
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
#include "apps/lib/oskar_dir.h"
#include <oskar_vis.h>
#include <oskar_measurement_set.h>

#include <oskar_cmath.h>
#include <cstdlib>

extern "C"
void oskar_vis_write_ms(const oskar_Vis* vis, const char* ms_path,
        int overwrite, int force_polarised, const char* run_log,
        size_t run_log_length, int* status)
{
    const oskar_Mem *x_metres, *y_metres, *z_metres, *vis_amp, *uu, *vv, *ww;
    oskar_Mem *amp_tb_;
    double dt_dump, t_start_sec, ref_freq_hz, chan_width, ra_rad, dec_rad;
    unsigned int num_stations, num_baselines, num_pols, num_channels, num_times;
    unsigned int precision, i, s1, s2, t, b, c, start_row;
    int *baseline_s1, *baseline_s2;
    oskar_MeasurementSet* ms;
    const void* amp;

    // Check all inputs.
    if (!vis || !ms_path || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    // Check if safe to proceed.
    if (*status) return;

    // Pull data from visibility structure.
    num_stations  = oskar_vis_num_stations(vis);
    num_baselines = oskar_vis_num_baselines(vis);
    num_pols      = oskar_vis_num_pols(vis);
    num_channels  = oskar_vis_num_channels(vis);
    num_times     = oskar_vis_num_times(vis);
    ra_rad        = oskar_vis_phase_centre_ra_deg(vis) * M_PI / 180.0;
    dec_rad       = oskar_vis_phase_centre_dec_deg(vis) * M_PI / 180.0;
    dt_dump       = oskar_vis_time_inc_sec(vis);
    t_start_sec   = oskar_vis_time_start_mjd_utc(vis) * 86400 + (dt_dump / 2);
    ref_freq_hz   = oskar_vis_freq_start_hz(vis);
    /* NOTE Should the chan_width be freq_inc or channel bandwidth?
     * (These are different numbers in general.) */
    chan_width    = oskar_vis_freq_inc_hz(vis);
    x_metres      = oskar_vis_station_x_offset_ecef_metres_const(vis);
    y_metres      = oskar_vis_station_y_offset_ecef_metres_const(vis);
    z_metres      = oskar_vis_station_z_offset_ecef_metres_const(vis);
    vis_amp       = oskar_vis_amplitude_const(vis);
    uu            = oskar_vis_baseline_uu_metres_const(vis);
    vv            = oskar_vis_baseline_vv_metres_const(vis);
    ww            = oskar_vis_baseline_ww_metres_const(vis);
    precision     = oskar_mem_precision(vis_amp);

    // Set channel width to be greater than 0, if it isn't already.
    // This is required for the Measurement Set to be valid.
    if (! (chan_width > 0.0))
        chan_width = 1.0;

    // Check if overwriting.
    if (overwrite)
    {
        if (oskar_dir_exists(ms_path))
        {
            // Try to overwrite.
            if (!oskar_dir_remove(ms_path))
            {
                *status = OSKAR_ERR_FILE_IO;
                return;
            }
        }

        // Create the Measurement Set.
        if (force_polarised) {
            ms = oskar_ms_create(ms_path, ra_rad, dec_rad, 4,
                    num_channels, ref_freq_hz, chan_width, num_stations, 0);
        }
        else {
            ms = oskar_ms_create(ms_path, ra_rad, dec_rad, num_pols,
                    num_channels, ref_freq_hz, chan_width, num_stations, 0);
        }

        // Set the station positions.
        if (oskar_mem_type(x_metres) == OSKAR_DOUBLE)
        {
            oskar_ms_set_station_coords_d(ms, num_stations,
                    oskar_mem_double_const(x_metres, status),
                    oskar_mem_double_const(y_metres, status),
                    oskar_mem_double_const(z_metres, status));
        }
        else if (oskar_mem_type(x_metres) == OSKAR_SINGLE)
        {
            oskar_ms_set_station_coords_f(ms, num_stations,
                    oskar_mem_float_const(x_metres, status),
                    oskar_mem_float_const(y_metres, status),
                    oskar_mem_float_const(z_metres, status));
        }
    }
    else
    {
        // Open the Measurement Set.
        ms = oskar_ms_open(ms_path);

        // Check the dimensions match.
        if (oskar_ms_num_channels(ms) != num_channels ||
                (!force_polarised && oskar_ms_num_pols(ms) != num_pols) ||
                (force_polarised && oskar_ms_num_pols(ms) != 4) ||
                oskar_ms_num_stations(ms) != num_stations)
        {
            *status = OSKAR_ERR_DIMENSION_MISMATCH;
            oskar_ms_close(ms);
            return;
        }

        // Check the reference frequencies match.
        if (fabs(oskar_ms_ref_freq_hz(ms) - ref_freq_hz) > 1e-10)
        {
            *status = OSKAR_ERR_REF_FREQ_MISMATCH;
            oskar_ms_close(ms);
            return;
        }

        // Check the phase centres are the same.
        if (fabs(oskar_ms_phase_centre_ra_rad(ms) - ra_rad) > 1e-10 ||
                fabs(oskar_ms_phase_centre_dec_rad(ms) - dec_rad) > 1e-10)
        {
            *status = OSKAR_ERR_PHASE_CENTRE_MISMATCH;
            oskar_ms_close(ms);
            return;
        }
    }

    // Evaluate baseline index arrays.
    baseline_s1 = (int*)malloc(num_baselines * sizeof(int));
    baseline_s2 = (int*)malloc(num_baselines * sizeof(int));
    for (s1 = 0, i = 0; s1 < num_stations; ++s1)
    {
        for (s2 = s1 + 1; s2 < num_stations; ++s2, ++i)
        {
            baseline_s1[i] = s1;
            baseline_s2[i] = s2;
        }
    }

    // Set size of the main table.
    start_row = oskar_ms_num_rows(ms);
    oskar_ms_set_num_rows(ms, start_row + num_times * num_baselines);

    // Add visibilities and u,v,w coordinates.
    amp = oskar_mem_void_const(vis_amp);
    if (force_polarised) {
        amp_tb_ = oskar_mem_create(precision | OSKAR_COMPLEX, OSKAR_CPU,
                num_baselines * num_channels * 4, status);
    }
    else {
        amp_tb_ = oskar_mem_create(precision | OSKAR_COMPLEX, OSKAR_CPU,
                num_baselines * num_channels * num_pols, status);
    }
    if (precision == OSKAR_DOUBLE)
    {
        double2* amp_tb;
        const double *uu_, *vv_, *ww_;
        uu_ = oskar_mem_double_const(uu, status);
        vv_ = oskar_mem_double_const(vv, status);
        ww_ = oskar_mem_double_const(ww, status);
        amp_tb = oskar_mem_double2(amp_tb_, status);
        for (t = 0; t < num_times; ++t)
        {
            unsigned int i_in, i_out, row = t * num_baselines;
            double vis_time = t_start_sec + (t * dt_dump);

            // Construct the amplitude data for the given time, baseline.
            if (num_pols == 1 && !force_polarised)
            {
                for (b = 0; b < num_baselines; ++b)
                {
                    for (c = 0; c < num_channels; ++c)
                    {
                        i_in = num_baselines * (num_times * c + t) + b;
                        i_out = num_channels * b + c;
                        amp_tb[i_out] = ((const double2*)amp)[i_in]; // I
                    }
                }
            }
            else
            {
                for (b = 0; b < num_baselines; ++b)
                {
                    for (c = 0; c < num_channels; ++c)
                    {
                        i_out = 4 * (num_channels * b + c);
                        if (force_polarised && num_pols == 1) {
                            i_in = num_baselines * (num_times * c + t) + b;
                            double2 vis_amp = ((const double2*)amp)[i_in];
                            amp_tb[i_out + 0] = vis_amp;         // XX
                            amp_tb[i_out + 1].x = 0.0;           // XY
                            amp_tb[i_out + 1].y = 0.0;           // XY
                            amp_tb[i_out + 2].x = 0.0;           // YX
                            amp_tb[i_out + 2].y = 0.0;           // YX
                            amp_tb[i_out + 3] = vis_amp;         // YY
                        }
                        else {
                            i_in = num_baselines * (num_times * c + t) + b;
                            double4c vis_amp = ((const double4c*)amp)[i_in];
                            amp_tb[i_out + 0] = vis_amp.a; // XX
                            amp_tb[i_out + 1] = vis_amp.b; // XY
                            amp_tb[i_out + 2] = vis_amp.c; // YX
                            amp_tb[i_out + 3] = vis_amp.d; // YY
                        }
                    }
                }
            }
            oskar_ms_write_all_for_time_d(ms, row + start_row,
                    num_baselines, &uu_[row], &vv_[row], &ww_[row],
                    (const double*)amp_tb, baseline_s1, baseline_s2,
                    oskar_vis_time_average_sec(vis), dt_dump, vis_time);
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
        for (t = 0; t < num_times; ++t)
        {
            unsigned int i_in, i_out, row = t * num_baselines;
            double vis_time = t_start_sec + (t * dt_dump);

            // Construct the amplitude data for the given time, baseline.
            if (num_pols == 1 && !force_polarised)
            {
                for (b = 0; b < num_baselines; ++b)
                {
                    for (c = 0; c < num_channels; ++c)
                    {
                        i_in = num_baselines * (num_times * c + t) + b;
                        i_out = num_channels * b + c;
                        amp_tb[i_out] = ((const float2*)amp)[i_in]; // I
                    }
                }
            }
            else
            {
                for (b = 0; b < num_baselines; ++b)
                {
                    for (c = 0; c < num_channels; ++c)
                    {
                        i_out = 4 * (num_channels * b + c);
                        if (force_polarised && num_pols == 1) {
                            i_in = num_baselines * (num_times * c + t) + b;
                            float2 vis_amp = ((const float2*)amp)[i_in];
                            amp_tb[i_out + 0] = vis_amp;        // XX
                            amp_tb[i_out + 1].x = 0.0;          // XY
                            amp_tb[i_out + 1].y = 0.0;          // XY
                            amp_tb[i_out + 2].x = 0.0;          // YX
                            amp_tb[i_out + 2].y = 0.0;          // YX
                            amp_tb[i_out + 3] = vis_amp;        // YY
                        }
                        else {
                            i_in = num_baselines * (num_times * c + t) + b;
                            float4c vis_amp = ((const float4c*)amp)[i_in];
                            amp_tb[i_out + 0] = vis_amp.a; // XX
                            amp_tb[i_out + 1] = vis_amp.b; // XY
                            amp_tb[i_out + 2] = vis_amp.c; // YX
                            amp_tb[i_out + 3] = vis_amp.d; // YY
                        }
                    }
                }
            }
            oskar_ms_write_all_for_time_f(ms, row + start_row,
                    num_baselines, &uu_[row], &vv_[row], &ww_[row],
                    (const float*)amp_tb, baseline_s1, baseline_s2,
                    oskar_vis_time_average_sec(vis), dt_dump, vis_time);
        }
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }

    // Add the settings.
    oskar_ms_add_settings(ms,
            oskar_mem_char_const(oskar_vis_settings_const(vis)),
            oskar_mem_length(oskar_vis_settings_const(vis)));

    // Add the run log.
    oskar_ms_add_log(ms, run_log, run_log_length);

    // Cleanup.
    oskar_mem_free(amp_tb_, status);
    free(baseline_s1);
    free(baseline_s2);
    oskar_ms_close(ms);
}
