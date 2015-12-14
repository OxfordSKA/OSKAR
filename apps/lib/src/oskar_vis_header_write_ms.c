/*
 * Copyright (c) 2015, The University of Oxford
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

#include "apps/lib/oskar_vis_header_write_ms.h"
#include "apps/lib/oskar_dir.h"

#include <oskar_cmath.h>
#include <oskar_measurement_set.h>
#include <oskar_vis_header.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_MeasurementSet* oskar_vis_header_write_ms(const oskar_VisHeader* hdr,
        const char* ms_path, int overwrite, int force_polarised, int* status)
{
    const oskar_Mem *x_metres, *y_metres, *z_metres;
    double ref_freq_hz, chan_width, ra_rad, dec_rad;
    int amp_type, autocorr, crosscorr, dir_exists;
    unsigned int num_stations, num_pols, num_channels;
    oskar_MeasurementSet* ms = 0;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Pull data from visibility structure. */
    amp_type     = oskar_vis_header_amp_type(hdr);
    num_stations = oskar_vis_header_num_stations(hdr);
    num_channels = oskar_vis_header_num_channels_total(hdr);
    ra_rad       = oskar_vis_header_phase_centre_ra_deg(hdr) * M_PI / 180.0;
    dec_rad      = oskar_vis_header_phase_centre_dec_deg(hdr) * M_PI / 180.0;
    ref_freq_hz  = oskar_vis_header_freq_start_hz(hdr);
    /* NOTE Should the chan_width be freq_inc or channel bandwidth?
     * (These are different numbers in general.) */
    chan_width   = oskar_vis_header_freq_inc_hz(hdr);
    x_metres     = oskar_vis_header_station_x_offset_ecef_metres_const(hdr);
    y_metres     = oskar_vis_header_station_y_offset_ecef_metres_const(hdr);
    z_metres     = oskar_vis_header_station_z_offset_ecef_metres_const(hdr);
    autocorr     = oskar_vis_header_write_auto_correlations(hdr);
    crosscorr    = oskar_vis_header_write_cross_correlations(hdr);
    num_pols     = oskar_type_is_matrix(amp_type) ? 4 : 1;

    /* Force creation of polarised output data if flag is set. */
    if (force_polarised) num_pols = 4;

    /* Set channel width to be greater than 0, if it isn't already.
     * This is required for the Measurement Set to be valid. */
    if (! (chan_width > 0.0))
        chan_width = 1.0;

    /* If directory doesn't exist, or if overwrite flag is set,
     * create a new one. */
    dir_exists = oskar_dir_exists(ms_path);
    if (!dir_exists || overwrite)
    {
        /* Remove any existing directory. */
        if (dir_exists && overwrite)
            oskar_dir_remove(ms_path);

        /* Create the Measurement Set. */
        ms = oskar_ms_create(ms_path, ra_rad, dec_rad, num_pols, num_channels,
                ref_freq_hz, chan_width, num_stations, autocorr, crosscorr);
        if (!ms)
        {
            *status = OSKAR_ERR_FILE_IO;
            return 0;
        }

        /* Set the station positions. */
        if (oskar_mem_type(x_metres) == OSKAR_DOUBLE)
        {
            oskar_ms_set_station_coords_d(ms, num_stations,
                    oskar_mem_double_const(x_metres, status),
                    oskar_mem_double_const(y_metres, status),
                    oskar_mem_double_const(z_metres, status));
        }
        else
        {
            oskar_ms_set_station_coords_f(ms, num_stations,
                    oskar_mem_float_const(x_metres, status),
                    oskar_mem_float_const(y_metres, status),
                    oskar_mem_float_const(z_metres, status));
        }

        /* Add the settings. */
        oskar_ms_add_settings(ms,
                oskar_mem_char_const(oskar_vis_header_settings_const(hdr)),
                oskar_mem_length(oskar_vis_header_settings_const(hdr)));
    }

    /* If directory already exists and we're not overwriting, open it. */
    else
    {
        /* Open the Measurement Set. */
        ms = oskar_ms_open(ms_path);
        if (!ms)
        {
            *status = OSKAR_ERR_FILE_IO;
            return 0;
        }

        /* Check the dimensions match. */
        if (oskar_ms_num_channels(ms) != num_channels ||
                oskar_ms_num_pols(ms) != num_pols ||
                oskar_ms_num_stations(ms) != num_stations)
        {
            *status = OSKAR_ERR_DIMENSION_MISMATCH;
            oskar_ms_close(ms);
            return 0;
        }

        /* Check the reference frequencies match. */
        if (fabs(oskar_ms_ref_freq_hz(ms) - ref_freq_hz) > 1e-10)
        {
            *status = OSKAR_ERR_REF_FREQ_MISMATCH;
            oskar_ms_close(ms);
            return 0;
        }

        /* Check the phase centres are the same. */
        if (fabs(oskar_ms_phase_centre_ra_rad(ms) - ra_rad) > 1e-10 ||
                fabs(oskar_ms_phase_centre_dec_rad(ms) - dec_rad) > 1e-10)
        {
            *status = OSKAR_ERR_PHASE_CENTRE_MISMATCH;
            oskar_ms_close(ms);
            return 0;
        }
    }

    return ms;
}

#ifdef __cplusplus
}
#endif
