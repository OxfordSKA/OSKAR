/*
 * Copyright (c) 2017, The University of Oxford
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

#include "imager/private_imager.h"
#include "imager/private_imager_read_dims.h"
#include "imager/oskar_imager.h"
#include "binary/oskar_binary.h"
#include "ms/oskar_measurement_set.h"
#include "vis/oskar_vis_header.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_read_dims_ms(oskar_Imager* h, const char* filename,
        int* status)
{
#ifndef OSKAR_NO_MS
    oskar_MeasurementSet* ms;
    int num_times, num_channels, num_stations, num_baselines, num_rows;
    if (*status) return;

    /* Read the header. */
    ms = oskar_ms_open(filename);
    if (!ms)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
    num_rows = (int) oskar_ms_num_rows(ms);
    num_stations = (int) oskar_ms_num_stations(ms);
    num_baselines = num_stations * (num_stations - 1) / 2;
    num_channels = (int) oskar_ms_num_channels(ms);
    num_times = num_rows / num_baselines;

    /* Check for irregular data and override time synthesis mode if required. */
    if (num_rows % num_baselines != 0)
    {
        oskar_log_warning(h->log,
                "Irregular data detected. Using full time synthesis.");
        oskar_imager_set_time_min_utc(h, 0.0);
        oskar_imager_set_time_max_utc(h, -1.0);
        oskar_imager_set_time_snapshots(h, 0);
    }

    /* Set visibility meta-data. */
    oskar_imager_set_vis_frequency(h,
            oskar_ms_freq_start_hz(ms),
            oskar_ms_freq_inc_hz(ms), num_channels);
    oskar_imager_set_vis_time(h,
            oskar_ms_time_start_mjd_utc(ms),
            oskar_ms_time_inc_sec(ms), num_times);
    oskar_ms_close(ms);
#else
    oskar_log_error(h->log, "OSKAR was compiled without Measurement Set support.");
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
#endif
}


void oskar_imager_read_dims_vis(oskar_Imager* h, const char* filename,
        int* status)
{
    oskar_Binary* vis_file;
    oskar_VisHeader* hdr;
    if (*status) return;

    /* Read the header. */
    vis_file = oskar_binary_create(filename, 'r', status);
    hdr = oskar_vis_header_read(vis_file, status);
    if (*status)
    {
        oskar_vis_header_free(hdr, status);
        oskar_binary_free(vis_file);
        return;
    }

    /* Set visibility meta-data. */
    oskar_imager_set_vis_frequency(h,
            oskar_vis_header_freq_start_hz(hdr),
            oskar_vis_header_freq_inc_hz(hdr),
            oskar_vis_header_num_channels_total(hdr));
    oskar_imager_set_vis_time(h,
            oskar_vis_header_time_start_mjd_utc(hdr),
            oskar_vis_header_time_inc_sec(hdr),
            oskar_vis_header_num_times_total(hdr));
    oskar_vis_header_free(hdr, status);
    oskar_binary_free(vis_file);
}


#ifdef __cplusplus
}
#endif
